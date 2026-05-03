from openai import OpenAI

def generate_text_response(prompt, model="gpt"):
    client = OpenAI()
    if model == "gemini":
        response = client.chat.completions.create(
            model="gemini-2.5-pro",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content, getattr(response.usage, "total_tokens", None) or 0
    else:
        response = client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
        )
        return response.output_text, getattr(response.usage, "total_tokens", None) or 0

def get_embedding(text):
    client = OpenAI()
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text, 
    )
    return response.data[0].embedding

def get_multiple_embeddings(texts):
    client = OpenAI()
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts, 
    )
    return [response.data[i].embedding for i in range(len(response.data))]