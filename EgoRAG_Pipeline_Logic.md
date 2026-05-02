# EgoRAG Pipeline Logic (Updated for M3-robot)

This document outlines the simplified memorization and reasoning logic for the updated EgoLife repository, now optimized for the M3-robot benchmark using GPT-5-mini.

## 1. Memorization Logic (`process_full_video.py`)

The memorization phase processes raw video frames into structured, searchable text summaries stored in a vector database.

### Step-by-Step Process:
1. **Frame Ingestion:** Reads sequential frame folders for a given video from `data/frames/<video_name>`.
2. **Clip-Level Captioning (MLLM):** 
   - Each folder represents a ~30-second clip.
   - The frames are passed to the MLLM (`gpt-5-mini` via `mllm_gpt.py`) along with a dense captioning prompt.
   - The MLLM generates a detailed factual caption describing actions, objects, and characters (specifically the "robot").
3. **Chunking & Summarization (LLM):**
   - The 30-second captions are grouped into 10-minute continuous chunks (e.g., 20 clips per chunk).
   - An LLM (`gpt-5-mini` via `llm.py`) summarizes these 20 captions into a single cohesive narrative paragraph focusing on key events and overarching actions.
4. **Database Storage:**
   - The 10-minute summaries are embedded using ChromaDB's default embedding function.
   - The embeddings, summary text, and metadata (`start_time` and `end_time` in seconds) are stored in a ChromaDB collection specific to the video.

---

## 2. Reasoning Logic (`RagAgent.py` & `main.py`)

The reasoning phase handles answering multiple-choice questions by intelligently retrieving and analyzing the stored memories.

### Step-by-Step Process:
1. **Query Parsing:** 
   - The system reads questions from the input JSON file.
   - It extracts the `question`, pre-defined `keywords` (if available), and the multiple-choice `options`.
2. **Keyword Extraction & Search Strategy:**
   - The LLM analyzes the question and options to formulate an optimal search query (keywords).
   - The system queries the Chroma database using these keywords to retrieve the **Top 10** most relevant 10-minute summaries.
3. **Coarse-to-Fine Document Selection:**
   - The retrieved Top 10 summaries are presented to the LLM.
   - The LLM acts as a reranker, selecting the single **best document index** that is most likely to contain the answer to the specific question.
4. **Evidence Extraction (Context Grounding):**
   - For the selected best document, the system retrieves its exact start and end times.
   - It reconstructs the detailed context by fetching the original underlying captions for that specific timeframe.
   - The LLM analyzes this detailed video context to explicitly extract concrete **"evidence cards"** (facts that directly support answering the question). If no evidence is found, it notes this.
5. **Final Answering:**
   - A final prompt is constructed combining the extracted evidence cards (or the raw captions if evidence extraction failed), the original question, and the multiple-choice options.
   - The LLM generates the final answer and selects the correct option (A, B, C, or D).
   - The system evaluates the predicted option against the ground truth to calculate accuracy.
