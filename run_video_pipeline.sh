#!/usr/bin/env bash
set -euo pipefail

# Run full pipeline per video. Videos can be processed in parallel.
# 0) Download required HF data folder (subtitles)
# 1) Download MP4
# 2) Extract frames with subtitles
# 3) Build memory (episodic + semantic)
# 4) Answer questions with main.py
# 5) Cleanup MP4 and frames

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Step 0: Ensure required data folder exists (data/ is gitignored)
echo "Preparing shared data folder..."
if python3 preprocessing/download_hf_folder.py; then
  echo "✓ Subtitles downloaded"
else
  echo "✗ Failed to download subtitles"
  exit 1
fi
echo ""

# Number of videos to process in parallel (set to 1 for sequential)
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-4}"

cleanup_video() {
  local video_name="$1"
  echo "Cleaning up files for ${video_name}..."
  rm -f "data/videos/${video_name}.mp4"
  rm -rf "data/frames/${video_name}"
}

process_one_video() {
  local video="$1"
  [[ -z "$video" ]] && return 0

  echo ""
  echo "[$(date +%H:%M:%S)] Processing video: ${video}"
  echo "============================================================"

  # Step 1: Download video
  if ! python3 preprocessing/download_hf_videos.py "$video"; then
    echo "✗ [${video}] Download failed"
    cleanup_video "$video"
    return 1
  fi

  # Step 2: Extract frames
  if ! python3 preprocessing/add_subtitles_and_extract_frames.py "$video"; then
    echo "✗ [${video}] Frame extraction failed"
    cleanup_video "$video"
    return 1
  fi

  # Step 3: Build memory
  echo "Generating memory for ${video}..."
  if python3 process_full_video.py --video_name "${video}"; then
    echo "✓ [${video}] Memory generated"
  else
    echo "✗ [${video}] Memory generation failed"
    cleanup_video "$video"
    return 1
  fi

  # Step 4: Answer questions with reason.py
  echo "Running reasoning for ${video}..."
  if python3 reason.py --name "${video}"; then
    echo "✓ [${video}] Reasoning complete"
  else
    echo "✗ [${video}] Reasoning failed"
    cleanup_video "$video"
    return 1
  fi

  # Step 5: Cleanup to free storage
  cleanup_video "$video"
  echo "✓ [${video}] Done (cleaned up)"
  return 0
}

if [[ "$#" -gt 0 ]]; then
  VIDEOS=("$@")
else
  if [[ ! -f "video_list.txt" ]]; then
    echo "video_list.txt not found. Pass video names as arguments."
    exit 1
  fi
  mapfile -t VIDEOS < "video_list.txt"
fi

echo "Processing ${#VIDEOS[@]} videos with max ${MAX_PARALLEL_JOBS} parallel jobs"
echo ""

# Run videos in parallel
i=0
while (( i < ${#VIDEOS[@]} )); do
  batch=()
  for ((j=0; j < MAX_PARALLEL_JOBS && i < ${#VIDEOS[@]}; j++)); do
    video="${VIDEOS[i]}"
    if [[ -n "$video" ]]; then
      batch+=("$video")
    fi
    (( i++ )) || true
  done
  if [[ ${#batch[@]} -gt 0 ]]; then
    for video in "${batch[@]}"; do
      ( process_one_video "$video" ) &
    done
    wait || true  # Continue even if some jobs failed
  fi
done

echo ""
echo "All video processing complete."
