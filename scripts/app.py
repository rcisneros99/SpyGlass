import streamlit as st
import os
import json
import asyncio
from ben import AudioProcessor
from dexter_doc_embeddings import get_texts_from_folder, get_embedding
from rafa import generate_counter_message
from openai import OpenAI
import numpy as np
from dotenv import load_dotenv
import base64
import sys

# Add the scripts directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Load environment variables
load_dotenv('.env.local')

# Initialize OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.openai.com/v1"
)

class IntelligenceApp:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.base_dir, "data")
        self.upload_dir = os.path.join(self.data_dir, "uploads")
        self.transcript_dir = os.path.join(self.data_dir, "transcripts")
        
        # Create necessary directories
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.transcript_dir, exist_ok=True)

    async def process_audio(self, uploaded_file):
        """Process uploaded audio file"""
        # Save uploaded file
        file_path = os.path.join(self.upload_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Process audio
        processor = AudioProcessor(
            input_dir=self.upload_dir,
            output_dir=self.transcript_dir,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        await processor.process_files()
        return True

    def generate_embeddings(self):
        """Generate embeddings for all transcripts"""
        texts, filenames = get_texts_from_folder(self.transcript_dir)
        
        if not texts:
            return False
            
        # Create a dictionary of filename to text content
        text_dict = dict(zip(filenames, texts))
        
        # Generate embeddings
        embeddings = {}
        for i, text in enumerate(texts):
            embeddings[filenames[i]] = get_embedding(text)
        
        # Save both embeddings and texts
        output = {
            "embeddings": embeddings,
            "texts": text_dict
        }
        
        embedding_path = os.path.join(self.transcript_dir, "embeddings.json")
        with open(embedding_path, "w") as f:
            json.dump(output, f)
        
        return True

    def query_documents(self, query):
        """Query the documents"""
        embedding_path = os.path.join(self.transcript_dir, "embeddings.json")
        
        try:
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
                relevant_texts = []
                messages = []
                for similarity, filename in top_docs:
                    relevant_texts.append(texts[filename])
                    messages.append({
                        "similarity": similarity,
                        "filename": filename,
                        "content": texts[filename]
                    })
                
                combined_text = "\n\n---\n\n".join(relevant_texts)
                
                # Improved prompt for better analysis
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": """You are an intelligence analyst for Ukraine analyzing intercepted Russian communications.
                        Your task is to:
                        1. Carefully analyze each message for threats, plans, intentions, details, or concerning information
                        2. Look for specific details about operations, movements, or strategic intentions
                        3. Consider both explicit and implicit threats
                        4. Identify any patterns or connections between messages
                        5. Highlight the most critical information first
                        
                        Provide a clear, concise analysis focusing on actionable intelligence based on the important messages. Do not include any messages that are not important to the question"""},
                        {"role": "user", "content": f"""Question: {query}

Intercepted Communications:
{combined_text}

Analyze these intercepted Russian communications and provide a detailed assessment focusing on the question."""}
                    ],
                    temperature=0.7,
                    max_tokens=500
                )
                answer = response.choices[0].message.content
                
                # Print for debugging
                print("\nAnalyzing Documents:")
                for msg in messages:
                    print(f"\nDocument (similarity: {msg['similarity']:.3f}):")
                    print(msg['content'][:200] + "...")
                print(f"\nAnalysis: {answer}")
                
                return {
                    "analysis": answer,
                    "messages": messages
                }
            
            return None
                
        except Exception as e:
            st.error(f"Error querying documents: {str(e)}")
            return None

def display_audio_file(file_path: str, file_type: str):
    """Display audio player for generated audio"""
    try:
        with open(file_path, "rb") as f:
            audio_bytes = f.read()
        
        audio_b64 = base64.b64encode(audio_bytes).decode()
        audio_tag = f'<audio controls><source src="data:audio/{file_type};base64,{audio_b64}" type="audio/{file_type}"></audio>'
        st.markdown(audio_tag, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error loading audio file: {str(e)}")

def main():
    st.set_page_config(layout="wide")
    st.title("🇺🇦 SpyGlass 🇺🇦")
    
    # Create two columns with better proportions
    side_panel, main_panel = st.columns([1, 2.5])
    
    # Initialize app
    app = IntelligenceApp()
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'counter_messages' not in st.session_state:
        st.session_state.counter_messages = {}
    if 'important_messages' not in st.session_state:
        st.session_state.important_messages = []
    if 'processed_messages' not in st.session_state:
        st.session_state.processed_messages = set()
    
    # Main Chat Panel (Load this first)
    with main_panel:
        st.header("Intelligence Chat")
        
        # Handle chat input first
        if prompt := st.chat_input("Ask a question about the intelligence"):
            # Add user message to session state
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Generate and store assistant response
            with st.spinner("Analyzing documents..."):
                response_context = app.query_documents(prompt)
                
                if response_context:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_context["analysis"],
                        "context": response_context
                    })
                else:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "No relevant information found."
                    })
        
        # Display chat messages from session state
        for i, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show source messages for assistant responses
                if message["role"] == "assistant" and "context" in message:
                    with st.expander("View source messages"):
                        for msg in message["context"]["messages"]:
                            st.markdown(f"""
                            **Relevance: {msg['similarity']:.2f}**
                            ```
                            {msg['content']}
                            ```
                            """)
                    
                    # Counter message generation UI
                    message_id = f"chat_msg_{i}"
                    if message_id not in st.session_state.counter_messages:
                        counter_intention = st.text_input(
                            "What should we make them believe?",
                            key=f"intention_{message_id}",
                            placeholder="e.g., 'Make them think we're retreating'"
                        )
                        
                        if st.button("Generate Counter Message", key=f"counter_btn_{message_id}"):
                            if not counter_intention:
                                st.warning("Please enter a counter message intention first.")
                            else:
                                try:
                                    with st.spinner("Generating counter message..."):
                                        counter_msg_english, counter_msg_ukrainian = generate_counter_message(
                                            message["context"]["analysis"],
                                            original_messages=message["context"]["messages"],
                                            intent=counter_intention
                                        )
                                        st.session_state.counter_messages[message_id] = {
                                            "english": counter_msg_english,
                                            "ukrainian": counter_msg_ukrainian,
                                            "intention": counter_intention
                                        }
                                        st.rerun()
                                except Exception as e:
                                    st.error(f"Error generating counter message: {str(e)}")
                    
                    if message_id in st.session_state.counter_messages:
                        with st.container():
                            st.markdown("---")
                            st.subheader("Counter Messages:")
                            st.markdown(f"🎯 Intention: {st.session_state.counter_messages[message_id]['intention']}")
                            st.markdown(f"🇺🇸 English: {st.session_state.counter_messages[message_id]['english']}")
                            st.markdown(f"🇺🇦 Ukrainian: {st.session_state.counter_messages[message_id]['ukrainian']}")
                            
                            st.subheader("Audio Files:")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("🇺🇸 English:")
                                display_audio_file("output_english.mp3", "mp3")
                            with col2:
                                st.markdown("🇺🇦 Ukrainian:")
                                display_audio_file("output_ukrainian.mp3", "mp3")

    # Side Panel
    with side_panel:
        # File uploader section
        with st.container():
            st.header("Upload Audio Files")
            uploaded_file = st.file_uploader("Choose an audio file", type=['mp3', 'wav'])
            
            if uploaded_file:
                with st.spinner("Processing audio file..."):
                    asyncio.run(app.process_audio(uploaded_file))
                    app.generate_embeddings()
                    # Clear important messages to force reload
                    st.session_state.important_messages = []
                st.success("File processed successfully!")
        
        # Critical Intel section
        st.markdown("---")
        critical_container = st.container()
        with critical_container:
            st.header("Critical Intel")
            
            # Load critical intel
            try:
                if len(st.session_state.important_messages) == 0:
                    with st.spinner("Loading critical intelligence..."):
                        response = app.query_documents("""Identify the most critical Russian communications that contain:
                        1. Specific military plans or operations
                        2. Important locations or movements
                        3. Strategic decisions or intentions
                        4. Time-sensitive information
                        5. Potential threats or vulnerabilities

                        Focus on messages that contain concrete details about operations, locations, or plans.
                        Ignore general chatter or non-actionable information.""")
                        
                        if response and isinstance(response, dict):
                            # Sort and filter messages
                            messages = response.get("messages", [])
                            filtered_messages = [
                                msg for msg in messages 
                                if msg["similarity"] > 0.6 and len(msg["content"].strip()) > 0
                            ]
                            
                            # Sort by similarity
                            sorted_messages = sorted(
                                filtered_messages,
                                key=lambda x: x["similarity"],
                                reverse=True
                            )[:10]  # Keep top 10
                            
                            if sorted_messages:
                                st.session_state.important_messages = sorted_messages
                                st.rerun()
                            else:
                                st.warning("No critical intelligence found in the documents.")
                        else:
                            st.warning("No documents available for analysis.")
                
                # Display critical intel
                for idx, msg in enumerate(st.session_state.important_messages):
                    with st.container():
                        # Message container with background color
                        is_processed = f"msg_{idx}" in st.session_state.counter_messages
                        bg_color = "#2d4744" if is_processed else "#1e1e1e"
                        
                        st.markdown(f"""
                            <div style="background-color: {bg_color}; padding: 10px; border-radius: 5px; margin: 5px 0;">
                                <p style="font-size: 0.9em;">{msg['content']}</p>
                                <p style="font-size: 0.8em; color: #888;">Relevance: {msg['similarity']:.2f}</p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        if not is_processed:
                            counter_intention = st.text_input(
                                "Counter message intention",
                                key=f"side_intention_{idx}",
                                placeholder="What should we make them believe?"
                            )
                            
                            if st.button("Generate Counter", key=f"side_btn_{idx}"):
                                if not counter_intention:
                                    st.warning("Please enter an intention first.")
                                else:
                                    try:
                                        with st.spinner("Generating counter message..."):
                                            counter_msg_english, counter_msg_ukrainian = generate_counter_message(
                                                msg['content'],
                                                intent=counter_intention
                                            )
                                            st.session_state.counter_messages[f"msg_{idx}"] = {
                                                "english": counter_msg_english,
                                                "ukrainian": counter_msg_ukrainian,
                                                "intention": counter_intention
                                            }
                                            st.rerun()
                                    except Exception as e:
                                        st.error(f"Error generating counter message: {str(e)}")
                        else:
                            st.markdown("✅ Counter message generated")
                            with st.expander("Show counter message"):
                                st.markdown(f"🎯 Intention: {st.session_state.counter_messages[f'msg_{idx}']['intention']}")
                                st.markdown(f"🇺🇸 {st.session_state.counter_messages[f'msg_{idx}']['english']}")
                                st.markdown(f"🇺🇦 {st.session_state.counter_messages[f'msg_{idx}']['ukrainian']}")
                                
                                st.subheader("Audio Files:")
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown("🇺🇸 English:")
                                    display_audio_file("output_english.mp3", "mp3")
                                with col2:
                                    st.markdown("🇺🇦 Ukrainian:")
                                    display_audio_file("output_ukrainian.mp3", "mp3")
                        
                        st.markdown("---")
            
            except Exception as e:
                st.error(f"Error loading critical intel: {str(e)}")

if __name__ == "__main__":
    main() 