import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from egorag.agents.RagAgent import RagAgent
from egorag.database.Chroma import Chroma
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Simplified EgoRAG Reasoning.")
    parser.add_argument("--name", type=str, required=True, help="Name of the video/database")
    parser.add_argument("--stage", nargs="+", default=["query", "answer"], help="Stages to run")
    args = parser.parse_args()

    NAME = args.name
    DB_NAME = NAME
    QUERY_JSON = "data/robot.json"

    # Initialize RAG agent
    rag_agent = RagAgent(
        database_t=Chroma(name=DB_NAME),
        name=NAME,
    )

    if "query" in args.stage:
        if not os.path.exists(QUERY_JSON):
            raise FileNotFoundError(f"Query JSON file not found: {QUERY_JSON}")
        
        print(f"Processing query file: {QUERY_JSON}")
        output_dir = "query_results"
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"{NAME}_results.json"
        output_filepath = os.path.join(output_dir, output_filename)
        
        with open(QUERY_JSON, "r", encoding="utf-8") as f:
            query_data = json.load(f)

        if NAME in query_data:
            query_list = query_data[NAME].get("qa_list", [])
        else:
            print(f"Warning: {NAME} not found in {QUERY_JSON}. Attempting to use as list.")
            query_list = query_data if isinstance(query_data, list) else []

        query_results = rag_agent.query_all(query_data=query_list)
        
        with open(output_filepath, "w", encoding="utf-8") as outfile:
            json.dump(query_results, outfile, ensure_ascii=False, indent=4)
        print(f"Query results saved to {output_filepath}")

    if "answer" in args.stage:
        print("answer stage")
        output_dir = "answer_results"
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"{NAME}_answers.json"
        output_filepath = os.path.join(output_dir, output_filename)

        if "query_results" not in locals():
            # Try to find the latest query result for this name
            query_res_dir = "query_results"
            latest_file = os.path.join(query_res_dir, f"{NAME}_results.json")
            if not os.path.exists(latest_file):
                # Fallback to check for files with timestamp prefix if the new format doesn't exist
                files = glob.glob(os.path.join(query_res_dir, f"*_{NAME}_results.json"))
                if not files:
                    raise ValueError(f"No previous query results found for {NAME}. Please run 'query' stage first.")
                latest_file = max(files, key=os.path.getctime)
            
            with open(latest_file, "r", encoding="utf-8") as f:
                query_results = json.load(f)
            print(f"Loaded query results from {latest_file}")

        answers = []
        for query_result in tqdm(query_results, desc="Answering queries"):
            try:
                query_range = query_result["result"]
                question = query_result["question"]
                formatted_options = query_result["formatted_options"]

                evidence_cards, caption_results = rag_agent.extract_evidence(question, query_range)
                ans, answer_option = rag_agent.get_answer(question, formatted_options, evidence_cards, caption_results=caption_results)

                answers.append({
                    "metadata": query_result["metadata"],
                    "model_answer": ans,
                    "model_option": answer_option,
                    "evidence_card": evidence_cards,
                })
            except Exception as e:
                print(f"Error processing query: {e}")
                continue

        acc = rag_agent.calculate_accuracy(answers, save_to_file=output_filepath)
        print(f"Accuracy for {NAME}: {acc}")

if __name__ == "__main__":
    import glob
    main()
