import argparse
import os
import json
from pathlib import Path
from tqdm import tqdm
import glob
import importlib

CAPTION_PROMPT = """You are an expert video captioner.
You will receive a short video segment represented by ordered frames. The input is actually consecutive images with no sound. The conversation is the subtitles of images.
Write a caption describing both the visual content and the audible content of the segment.

Guidelines:
- Describe visible actions, people, objects, and environment.
- Keep the caption factual and neutral.
- Output only the final caption text.
- IMPORTANT: Output the description separated by sentences (e.g. split by periods)."""

def process_full_video(video_name: str, model: str):
    # Dynamically load the correct mllm module
    if model == "gemini":
        mllm = importlib.import_module("utils.mllm_gemini")
    else:
        mllm = importlib.import_module("utils.mllm_gpt")
        
    frames_dir = Path(f"data/frames/{video_name}")
    if not frames_dir.exists() or not frames_dir.is_dir():
        print(f"Frames directory not found: {frames_dir}")
        return

    output_dir = Path(f"data/memorization_{model}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{video_name}.json"
    
    image_folders = sorted(
        [str(folder) for folder in frames_dir.iterdir() if folder.is_dir()],
        key=lambda x: int(Path(x).name)
    )
    
    memorization_data = {}
    total_tokens = 0
    
    for folder in tqdm(image_folders, desc=f"Captioning clips ({model})"):
        clip_id = Path(folder).name
        current_images = sorted(
            glob.glob(f"{folder}/*.jpg"),
            key=lambda p: int(Path(p).stem) if Path(p).stem.isdigit() else p,
        )
        if not current_images:
            continue
            
        messages = mllm.generate_messages(current_images, CAPTION_PROMPT)
        try:
            response, tokens = mllm.get_response(messages)
            total_tokens += tokens
            
            # Split the response into sentences
            sentences = [s.strip() for s in response.replace('\n', '.').split('.') if s.strip()]
            memorization_data[clip_id] = sentences
            
        except Exception as e:
            print(f"MLLM failed for clip {clip_id}: {e}")
            continue

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(memorization_data, f, ensure_ascii=False, indent=4)
        
    print(f"Memory generation complete for {video_name} using {model}! Total tokens used: {total_tokens}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_name", type=str, required=True, help="Name of the video to process")
    parser.add_argument("--model", type=str, choices=["gpt", "gemini"], default="gpt", help="Model to use")
    args = parser.parse_args()
    
    process_full_video(video_name=args.video_name, model=args.model)
