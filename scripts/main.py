import os
import json
import openai
import random
from datetime import datetime
from scripts.ben import AudioProcessor
from scripts.dexter_doc_embeddings import get_texts_from_folder, get_embedding, query_with_gpt, find_most_relevant, get_embeddings_batch
from scripts.rafa import generate_counter_message
import asyncio
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv('.env.local')

# Configure OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

class IntelligenceSystem:
    def __init__(self):
        print("\n=== Initializing Intelligence System ===")
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.base_dir, "data")
        self.upload_dir = os.path.join(self.data_dir, "uploads")
        self.transcript_dir = os.path.join(self.data_dir, "transcripts")
        
        # Create necessary directories
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.transcript_dir, exist_ok=True)
        
        # Initialize OpenAI client
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
            
        print("✓ Directories created/verified")
        print(f"✓ Found {len(os.listdir(self.transcript_dir))} existing transcripts")
        print("✓ System initialized")

    async def process_audio(self):
        """Step 1: Process audio using Ben's module"""
        print("\n=== STEP 1: AUDIO PROCESSING ===")
        
        # Get random file from uploads
        audio_files = [f for f in os.listdir(self.upload_dir) if f.endswith(('.mp3', '.wav'))]
        if not audio_files:
            print("⚠️ No audio files found in uploads directory")
            return False
            
        selected_file = random.choice(audio_files)
        print(f"Selected file for processing: {selected_file}")
        
        processor = AudioProcessor(
            input_dir=self.upload_dir,
            output_dir=self.transcript_dir,
            openai_api_key=self.openai_api_key
        )
        
        try:
            print("\nInput to ben.py:")
            print(f"- Audio file: {os.path.join(self.upload_dir, selected_file)}")
            print(f"- Output directory: {self.transcript_dir}")
            
            await processor.process_files()
            
            print("\nOutput from ben.py:")
            print(f"- Transcript files created in: {self.transcript_dir}")
            return True
        except Exception as e:
            print(f"⚠️ Error processing audio: {str(e)}")
            return False

    def generate_embeddings(self):
        """Step 2: Generate embeddings using Dexter's module"""
        print("\n=== STEP 2: GENERATING EMBEDDINGS ===")
        
        try:
            print("\nInput to dexter_doc_embeddings.py:")
            print(f"- Transcript directory: {self.transcript_dir}")
            
            texts, filenames = get_texts_from_folder(self.transcript_dir)
            
            if not texts:
                print("⚠️ No transcripts found to process")
                return False
                
            print(f"Found {len(texts)} transcripts to process")
            
            # Create a dictionary of filename to text content
            text_dict = dict(zip(filenames, texts))
            
            # Generate embeddings in batches
            print("Generating embeddings...")
            all_embeddings = get_embeddings_batch(texts)
            embeddings = dict(zip(filenames, all_embeddings))
            
            # Save both embeddings and texts
            output = {
                "embeddings": embeddings,
                "texts": text_dict
            }
            
            embedding_path = os.path.join(self.transcript_dir, "embeddings.json")
            with open(embedding_path, "w") as f:
                json.dump(output, f)
                
            print("\nOutput from dexter_doc_embeddings.py:")
            print(f"- Generated embeddings for {len(embeddings)} documents")
            print(f"- Saved to: {embedding_path}")
            return True
            
        except Exception as e:
            print(f"⚠️ Error generating embeddings: {str(e)}")
            return False

    def query_documents(self, query):
        """Step 3: Query the documents using embeddings"""
        print(f"\n=== STEP 3: QUERYING DOCUMENTS ===")
        print(f"Question: '{query}'")
        
        embedding_path = os.path.join(self.transcript_dir, "embeddings.json")
        
        try:
            print("\nInput to dexter_question_embeddings.py:")
            print(f"- Question: {query}")
            
            # Load embeddings and texts
            with open(embedding_path, 'r') as f:
                data = json.load(f)
                embeddings = data["embeddings"]
                texts = data["texts"]
            
            # Get query embedding
            query_embedding = get_embedding(query)
            
            # Find most relevant documents
            similarities = []
            for filename, doc_embedding in embeddings.items():
                similarity = np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                )
                similarities.append((similarity, filename))
            
            # Get top 3 most similar documents
            top_docs = sorted(similarities, reverse=True)[:3]
            
            if top_docs:
                print("\nFound relevant documents:")
                relevant_texts = []
                for similarity, filename in top_docs:
                    print(f"\n- {filename} (similarity: {similarity:.3f})")
                    print(f"Content: {texts[filename][:200]}...")
                    relevant_texts.append(texts[filename])
                
                combined_text = "\n\n---\n\n".join(relevant_texts)
                
                print("\nQuerying GPT for analysis...")
                # Create a new OpenAI client with standard API URL
                gpt_client = OpenAI(
                    api_key=self.openai_api_key,
                    base_url="https://api.openai.com/v1"
                )
                
                response = gpt_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": """You are an intelligence analyst for Ukraine. 
                        Your task is to analyze intercepted Russian communications and extract actionable intelligence.
                        Focus on identifying:
                        - Planned actions or operations
                        - Locations and movements
                        - Strategic intentions
                        If you find relevant information, summarize it concisely. 
                        If you don't find specific information, say "No specific plans identified in these communications." """},
                        {"role": "user", "content": f"Analyze these intercepted communications and tell me about their plans:\n\n{combined_text}"}
                    ]
                )
                answer = response.choices[0].message.content
                
                print("\nGPT Analysis:")
                print(answer)
                
                # Only return if we found actual information
                if "No specific plans" not in answer:
                    # Return both the analysis and the original messages
                    context = {
                        "analysis": answer,
                        "messages": [
                            {"similarity": sim, "filename": fname, "content": texts[fname]}
                            for sim, fname in top_docs
                        ]
                    }
                    return combined_text, context
                else:
                    print("\nNo actionable intelligence found")
                    return None, None
            else:
                print("⚠️ No relevant documents found")
                return None, None
                
        except Exception as e:
            print(f"⚠️ Error querying documents: {str(e)}")
            return None, None

    def generate_counter(self, message_context):
        """Step 4: Generate counter message"""
        print(f"\n=== STEP 4: GENERATING COUNTER MESSAGE ===")
        
        try:
            print("\nInput to rafa.py:")
            print(f"- Analysis: {message_context['analysis']}")
            print("\nRelevant intercepted messages:")
            for msg in message_context['messages']:
                print(f"\n- Message (similarity: {msg['similarity']:.3f}):")
                print(f"  {msg['content'][:200]}...")
            
            counter_msg_english, counter_msg_ukrainian = generate_counter_message(
                message_context['analysis'],
                original_messages=message_context['messages']
            )
            
            print("\nOutput from rafa.py:")
            print(f"- English counter message: {counter_msg_english}")
            print(f"- Ukrainian counter message: {counter_msg_ukrainian}")
            
            # Ask for user confirmation
            confirmation = input("\nGenerate audio for this message? (y/n): ")
            
            if confirmation.lower() == 'y':
                print("Generating audio files...")
                # Audio files are already generated by generate_counter_message
                print("✓ Counter message audio generated")
                return True
            else:
                print("Audio generation cancelled")
                return False
                
        except Exception as e:
            print(f"⚠️ Error generating counter message: {str(e)}")
            return False

async def main():
    # Initialize the system
    system = IntelligenceSystem()
    
    # Step 1: Process any new audio files
    await system.process_audio()
    
    # Step 2: Generate embeddings
    system.generate_embeddings()
    
    # Step 3: Query example
    query = "What are their next planned actions?"
    relevant_text, answer = system.query_documents(query)
    
    # Step 4: Generate counter message if we found relevant information
    if relevant_text and answer:
        system.generate_counter(answer)

    print("\n=== Processing Complete ===")

if __name__ == "__main__":
    asyncio.run(main()) 