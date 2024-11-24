import os
import json
import sys
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np

# Load environment variables
load_dotenv('.env.local')

# Initialize OpenAI client for embeddings
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.openai.com/v1"  # Use standard OpenAI API for embeddings
)

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
    """Get embedding for a single text"""
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

def get_embeddings_batch(texts):
    """Get embeddings for multiple texts"""
    embeddings = []
    for text in texts:
        embedding = get_embedding(text)
        embeddings.append(embedding)
    return embeddings

def query_with_gpt(query: str, relevant_text: str) -> str:
    """Query GPT with the relevant text context"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an intelligence analyst. Answer questions based on the provided intelligence documents."},
            {"role": "user", "content": f"Based on this intelligence document:\n\n{relevant_text}\n\nQuestion: {query}"}
        ]
    )
    return response.choices[0].message.content

def find_most_relevant(query: str, embeddings: dict, texts: dict, top_k=3):
    """Find the most relevant documents and combine them for context"""
    query_embedding = get_embedding(query)
    
    similarities = []
    for filename, doc_embedding in embeddings.items():
        similarity = np.dot(query_embedding, doc_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
        )
        similarities.append((similarity, filename))
    
    # Get top k most similar documents
    top_docs = sorted(similarities, reverse=True)[:top_k]
    
    # Combine the relevant texts
    relevant_texts = []
    for _, filename in top_docs:
        relevant_texts.append(texts[filename])
    
    return "\n\n---\n\n".join(relevant_texts)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide a folder path")
        sys.exit(1)
        
    folder = sys.argv[1]
    texts, filenames = get_texts_from_folder(folder)
    
    print(f"Processing {len(texts)} documents...")
    
    # Create a dictionary of filename to text content
    text_dict = dict(zip(filenames, texts))
    
    # Generate embeddings in batches
    print("Generating embeddings...")
    all_embeddings = get_embeddings_batch(texts)
    
    # Create embeddings dictionary
    embeddings = dict(zip(filenames, all_embeddings))
    
    # Save both embeddings and texts
    output = {
        "embeddings": embeddings,
        "texts": text_dict
    }
    
    output_file = os.path.join(folder, "embeddings.json")
    with open(output_file, "w") as f:
        json.dump(output, f)
    
    print(f"Saved embeddings to {output_file}")