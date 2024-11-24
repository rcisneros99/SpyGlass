import openai
import sys
import os
import json
openai.api_key = "OPEN_AI_KEY"

question = sys.argv[1]

def get_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    embedding = response['data'][0]['embedding']
    return embedding

question_embedding = {}
question_embedding[question] = get_embedding(question)


output_file = os.path.join("question_embedding.json")
with open(output_file, "w") as f:
    json.dump(question_embedding, f)
