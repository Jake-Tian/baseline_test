import json
import os
import re
from typing import Any, Dict, List, Optional
import random
from egorag.database.Chroma import Chroma
from egorag.models.llm import generate_text_response
from egorag.utils.util import extract_single_option_answer
from egorag.utils.prompts import *
from tqdm import tqdm

def call_gpt4(prompt, system_message="You are an AI assistant.", temperature=0.1):
    messages = [{"role": "system", "content": system_message}, {"role": "user", "content": prompt}]
    res, _ = generate_text_response(messages)
    return res

class RagAgent:
    def __init__(
        self,
        database_t: Optional[Chroma] = None,
        name: str = "NULL",
    ):
        self.database_t = database_t
        self.name = name

    def calculate_accuracy(self, answers, save_to_file=None):
        total_count = 0
        correct_count = 0
        results = []

        for answer in tqdm(answers, desc="Evaluating answers"):
            correct_answer = answer["metadata"]["answer"]
            model_prediction = answer["model_answer"]
            question = answer["metadata"]["question"]

            total_count += 1
            
            # Use GPT-4 for semantic evaluation of open-ended answers
            eval_prompt = f"""Evaluate the correctness of the predicted answer compared to the ground truth.
Question: {question}
Ground Truth: {correct_answer}
Predicted Answer: {model_prediction}

Is the predicted answer semantically correct and contains the same core information as the ground truth? Output only 'Yes' or 'No'."""
            
            eval_res = call_gpt4(eval_prompt).strip().lower()
            is_correct = "yes" in eval_res

            correct_count += 1 if is_correct else 0
            answer["is_correct"] = is_correct
            results.append(answer)
            
        accuracy = correct_count / total_count if total_count > 0 else 0

        if save_to_file:
            with open(save_to_file, "w", encoding="utf-8") as f:
                json.dump({"accuracy": accuracy, "results": results}, f, ensure_ascii=False, indent=4)

        return accuracy

    def process_query_results(self, raw_results):
        processed_results = []
        if not raw_results["ids"] or not raw_results["ids"][0]:
            return processed_results

        ids = raw_results["ids"][0]
        documents = raw_results["documents"][0]
        metadatas = raw_results["metadatas"][0]
        distances = raw_results["distances"][0]

        combined_results = [
            (
                ids[i],
                documents[i],
                metadatas[i].get("end_time", 0),
                distances[i],
            )
            for i in range(len(ids))
        ]

        combined_results.sort(key=lambda x: x[2])
        for id, document, end_time, distance in combined_results:
            results = self.database_t.get_caption(id=id, n_result=1)
            expand_documents = results["documents"]
            processed_result = {
                "id": id,
                "document": expand_documents,
                "end_time": end_time,
                "distance": distance,
            }
            processed_results.append(processed_result)

        return processed_results

    def query(self, query_dict):
        question, query, formatted_options = self.parse_query(query_dict)
        
        prompt = f"Question: {question} \nOptions: \n{formatted_options}\nOutput:"
        keyword_raw = call_gpt4(prompt=prompt, system_message=query_prompt, temperature=0.1)
        
        keyword_pattern = r"Keywords:\s*\[([^\]]+)\]"
        keyword_match = re.search(keyword_pattern, keyword_raw)
        keywords = keyword_match.group(1).strip() if keyword_match else question
        
        # Directly retrieve from Chroma (10 min summaries)
        raw_results = self.database_t.custom_query(
            query_texts=[keywords],
            n_results=10, # Retrieve top 10 relevant summaries
            filter_first=False,
        )
        
        all_query_results = self.process_query_results(raw_results)
        
        # Select the best document id using GPT-4
        prompt_str = f"Given the question: '{question}', and the following retrieved documents:\n"
        for i, res in enumerate(all_query_results):
            prompt_str += f"[{i}]: {res['document']}\n"
        prompt_str += "Which document best answers the question? Output only the index number."
        
        idx_res = call_gpt4(prompt=prompt_str)
        try:
            best_idx = int(re.search(r'\d+', idx_res).group())
            best_id = all_query_results[best_idx]["id"]
        except:
            best_id = all_query_results[0]["id"] if all_query_results else None
            
        if best_id:
            results = self.database_t.get_caption(id=best_id, n_result=1)
            best_result = results["item"]
        else:
            best_result = {"ids": [], "documents": [], "metadatas": []}

        query_result = [
            {
                "query_range": None,
                "docs": best_result,
                "extract_keywords": keywords,
                "filter_id": best_id,
            }
        ]

        return query_result, question, formatted_options

    def single_query(self, query_dict):
        return self.query(query_dict)

    def query_all(self, query_data):
        all_query_results = []
        for query_dict in tqdm(query_data, desc="Processing queries"):
            try:
                query_result, question, formatted_options = self.single_query(query_dict)
                all_query_results.append(
                    {
                        "metadata": query_dict,
                        "result": query_result,
                        "question": question,
                        "formatted_options": formatted_options,
                    }
                )
            except Exception as e:
                print(f"error {e}")
                continue
        return all_query_results

    def parse_query(self, data_dict):
        question = data_dict.get("question", "UNKNOWN_QUERY")
        query = data_dict.get("keywords", "UNKNOWN_QUERY")
        options = []
        for letter in ["a", "b", "c", "d"]:
            choice_key = f"choice_{letter}"
            if choice_key in data_dict:
                options.append(f"{letter.upper()}. {data_dict[choice_key]}")
        formatted_options = "\n".join(options)
        return question, query, formatted_options

    def extract_evidence(self, question, query_range):
        def parse_evidence_output(response):
            if "I can't provide evidence." in response:
                return {"status": False}
            match = re.search(r"I can provide evidence\. Evidence: (.+)", response)
            if match:
                return {"status": True, "information": match.group(1).strip()}
            return {"status": False}

        doc_ids = []
        evidence = []
        caption_results = {}
        single_range = query_range[0]["docs"]
        
        for index, id in enumerate(single_range["ids"]):
            start_time = single_range["metadatas"][index].get("start_time", 0)
            end_time = single_range["metadatas"][index].get("end_time", 0)

            all_result = self.database_t.get_caption(id=id, n_result=1)
            full_caption = " ".join(all_result["documents"])
            video_caption = f"Video Time: {start_time} to {end_time} seconds.\nVideo Content: {full_caption}"
            
            caption_key = f"video{index+1}"
            caption_results[caption_key] = video_caption
            doc_ids.append(single_range["ids"][index])
            
            text_prompt = f"Given the video context:\n{video_caption}\nAnswer the question: {question}\nDoes this contain evidence?"
            response = call_gpt4(text_prompt)
            evidence.append(parse_evidence_output(response))

        evidence_cards = []
        for index, id in enumerate(doc_ids):
            evidence_card = {"time": id}
            if evidence[index]["status"]:
                evidence_card["evidence_info"] = evidence[index]["information"]
            else:
                evidence_card["evidence_info"] = "No evidence information."
            evidence_cards.append(evidence_card)
            
        return evidence_cards, caption_results

    def get_answer(self, question, formatted_options, evidence_cards, caption_results):
        combined_string = "\n\n".join(
            f"time: {d.get('time', 'N/A')}\n"
            f"evidence_info: {d.get('evidence_info', 'No evidence info')}\n"
            for d in evidence_cards
        )

        prompt_with_evidence = f"Evidence:\n{combined_string}\nQuestion: {question}\nOptions:\n{formatted_options}\nAnswer:"
        prompt_without_evidence = f"Captions:\n{caption_results}\nQuestion: {question}\nOptions:\n{formatted_options}\nAnswer:"
        
        if not evidence_cards:
            ans = call_gpt4(prompt=prompt_without_evidence, temperature=0.1)
        else:
            ans = call_gpt4(prompt=prompt_with_evidence, temperature=0.1)
            
        if ans == "error":
            ans = "answer error"
            answer_option = None
        else:
            if formatted_options:
                answer_option = extract_single_option_answer(ans)
            else:
                answer_option = ans

        return ans, answer_option
