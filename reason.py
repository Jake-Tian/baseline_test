import argparse
import os
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np

from egorag.models.llm import generate_text_response, get_multiple_embeddings, get_embedding

def verify_answer(generated_answer: str, ground_truth: str) -> bool:
    prompt = f"Given the ground truth answer: '{ground_truth}', is the following generated answer correct and semantically equivalent? Generated answer: '{generated_answer}'. Output ONLY 'yes' or 'no'."
    try:
        response, _ = generate_text_response(prompt)
        return 'yes' in response.lower()
    except Exception as e:
        print(f"LLM verification failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Simplified EgoRAG Reasoning.")
    parser.add_argument("--name", type=str, required=True, help="Name of the video")
    args = parser.parse_args()

    video_name = args.name
    memory_path = Path(f"data/memorization/{video_name}.json")
    query_json_path = "data/robot.json"
    
    if not memory_path.exists():
        print(f"Memory file not found: {memory_path}. Please run process_full_video.py first.")
        return
        
    with open(memory_path, "r", encoding="utf-8") as f:
        memorization_data = json.load(f)
        
    # Flatten memory into a list of sentences
    all_sentences = []
    
    for clip_id, sentences in memorization_data.items():
        for sentence in sentences:
            all_sentences.append(sentence)

    print(f"Loaded {len(all_sentences)} sentences from memory.")

    sentence_embeddings = []
    if len(all_sentences) > 0:
        batch_size = 100
        print("Embedding memory sentences...")
        for i in tqdm(range(0, len(all_sentences), batch_size)):
            batch_sentences = all_sentences[i:i+batch_size]
            try:
                embeddings = get_multiple_embeddings(batch_sentences)
                sentence_embeddings.extend(embeddings)
            except Exception as e:
                print(f"Failed to embed batch {i}: {e}")
                # Fill with zeros or fallback
                sentence_embeddings.extend([[0.0] * 1536] * len(batch_sentences))
        
        sentence_embeddings = np.array(sentence_embeddings)

    # Load queries
    with open(query_json_path, "r", encoding="utf-8") as f:
        query_data = json.load(f)
        
    if video_name not in query_data:
        print(f"Video {video_name} not found in {query_json_path}")
        return
        
    qa_list = query_data[video_name].get("qa_list", [])
    if not qa_list:
        print(f"No queries found for {video_name}")
        return

    output_dir = Path("data/reason")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{video_name}.json"
    
    results = {}
    
    print("Processing queries...")
    for qa in tqdm(qa_list, desc="Reasoning"):
        question = qa["question"]
        question_id = qa.get("question_id", question)
        ground_truth = qa["answer"]
        
        retrieved_sentences = []
        if len(all_sentences) > 0:
            try:
                # Retrieve top-k (k=50)
                q_emb = np.array(get_embedding(question))
                
                # Cosine similarity
                similarities = np.dot(sentence_embeddings, q_emb) / (np.linalg.norm(sentence_embeddings, axis=1) * np.linalg.norm(q_emb))
                
                top_k = min(50, len(all_sentences))
                top_indices = np.argsort(similarities)[-top_k:][::-1]
                retrieved_sentences = [all_sentences[idx] for idx in top_indices]
            except Exception as e:
                print(f"Failed to query sentences: {e}")
            
        # Put into prompt combined with query
        context = "\n".join([f"- {s}" for s in retrieved_sentences])
        prompt = f"Based on the following retrieved memories from a video:\n{context}\n\nAnswer the question: {question}"
        
        # Call LLM first time: generate answer
        try:
            generated_answer, _ = generate_text_response(prompt)
        except Exception as e:
            print(f"LLM failed to generate answer: {e}")
            generated_answer = ""
        
        # Call LLM second time: verify
        is_correct = verify_answer(generated_answer, ground_truth)
        
        results[question_id] = {
            "question": question,
            "ground_truth": ground_truth,
            "generated_answer": generated_answer,
            "is_correct": is_correct,
            "retrieved_context": retrieved_sentences
        }
        
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
    print(f"Reasoning complete for {video_name}! Results saved to {output_path}")

if __name__ == "__main__":
    main()
