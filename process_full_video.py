import argparse
import os
from pathlib import Path
from tqdm import tqdm
import glob
from egorag.database.Chroma import Chroma
from egorag.models.mllm_gpt import generate_messages as mllm_generate_messages, get_response as mllm_get_response
from egorag.models.llm import generate_text_response

CAPTION_PROMPT = """You are an expert video captioner.
You will receive a short video segment represented by ordered frames. The input is actually consecutive images with no sound. The conversation is the subtitles of images.
Write a caption describing both the visual content and the audible content of the segment.

Guidelines:
- Describe visible actions, people, objects, and environment.
- The camera is held by a robot ("robot"). The robot wears black gloves and has no visible face. Explicitly describe the robot's visible actions (e.g., its hands interacting with objects) and any speech.
- Always use characters' real names if they are provided in the subtitles (e.g., Anna, Susan). Use these names consistently to refer to the characters performing actions.
- If a character's name is not provided in the subtitles, describe them distinctively or use a placeholder name like person_1.
- Include relevant speech, sounds, or audio events based on subtitles.
- Keep the caption factual and neutral.
- Do not mention frames, timestamps, or that the input came from frames.
- Avoid speculation about emotions or intentions unless clearly visible or stated in speech.

Output only the final caption text."""

def summarize_captions(captions: list[str]) -> str:
    prompt = f"Summarize the following series of video captions into a single cohesive 10-minute summary paragraph. Focus on the main events, key actions, and overarching narrative.\n\n"
    prompt += "Captions:\n"
    for i, cap in enumerate(captions):
        prompt += f"Clip {i+1}: {cap}\n"
    prompt += "\nSummary:"
    response, tokens = generate_text_response(prompt)
    return response

def process_full_video(video_name: str):
    frames_dir = Path(f"data/frames/{video_name}")
    if not frames_dir.exists() or not frames_dir.is_dir():
        print(f"Frames directory not found: {frames_dir}")
        return

    # Initialize Chroma database
    db = Chroma(name=video_name)
    
    print(f"Generating 30sec captions for {video_name}...")
    image_folders = sorted(
        [str(folder) for folder in frames_dir.iterdir() if folder.is_dir()],
        key=lambda x: int(Path(x).name)
    )
    
    caption_data_30sec = []
    
    for folder in tqdm(image_folders, desc="Captioning clips"):
        clip_id = Path(folder).name
        current_images = sorted(
            glob.glob(f"{folder}/*.jpg"),
            key=lambda p: int(Path(p).stem) if Path(p).stem.isdigit() else p,
        )
        if not current_images:
            continue
            
        messages = mllm_generate_messages(current_images, CAPTION_PROMPT)
        try:
            response, _ = mllm_get_response(messages)
        except Exception as e:
            print(f"MLLM failed for clip {clip_id}: {e}")
            continue
            
        caption_data_30sec.append({
            "text": response.strip(),
            "clip_id": int(clip_id),
            "start_time": int(clip_id) * 30, # assuming 30 seconds per clip
            "end_time": (int(clip_id) + 1) * 30
        })

    print(f"Generating 10-minute summaries and saving to Chroma...")
    # 10 minutes = 20 clips (if 30sec each)
    chunk_size = 20
    
    for i in range(0, len(caption_data_30sec), chunk_size):
        chunk = caption_data_30sec[i:i+chunk_size]
        texts = [c['text'] for c in chunk]
        summary = summarize_captions(texts)
        
        start_time = chunk[0]['start_time']
        end_time = chunk[-1]['end_time']
        
        # Save to Chroma DB
        entry_id = f"10min_{start_time}_{end_time}"
        
        metadata = {
            "start_time": start_time,
            "end_time": end_time,
            "video_path": video_name
        }
        
        # Embed and insert
        embeddings = db.embedding_function([summary])
        db.collection.add(
            ids=[entry_id],
            documents=[summary],
            metadatas=[metadata],
            embeddings=embeddings
        )

    print(f"Database generation complete for {video_name}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_name", type=str, required=True, help="Name of the video to process")
    args = parser.parse_args()
    
    process_full_video(video_name=args.video_name)
