I copy the EgoLife code and I'm going to modify it to a new version which is much simpler and used to test the baselines. 

1. process_full_video.py: process the video clip one by one and save it to a json file in data/ with the format, key is the clip_id and value is a list of generated memory of the clip seperated by sentences. For each clip only call mllm once. The json should be saved in data/memorization/ with the name of each video. Don't forget to monitor the token consumption. 

2. reason.py: First use chroma_db to generate the embedding for memory sentence by sentence. Then apply the top-k (k=50) retrieval based on the query and then put into the combined with the query. For each question only call llm twice, one is to generate the answer, the other one is to verify with the ground-truth answer. The json should be saved in data/reason with the name of each video as the format of dictionaries. 

Make the code as simple as possible. Remove all the unused files. 