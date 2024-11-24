import openai
import sys
import os
import json
openai.api_key = "OPEN_AI_KEY"

folder = sys.argv[1]

def get_texts_from_folder(folder):
    texts = []
    filenames = []
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            filenames.append(filename)
            with open(os.path.join(folder, filename), "r") as file:
                texts.append(file.read())
    return texts, filenames

def get_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    embedding = response['data'][0]['embedding']
    return embedding

texts,filenames = get_texts_from_folder(folder)


embeddings = {}
for i,file in enumerate(filenames):
    embeddings[file] = get_embedding(texts[i])


output_file = os.path.join(folder, "embeddings.json")
with open(output_file, "w") as f:
    json.dump(embeddings, f)