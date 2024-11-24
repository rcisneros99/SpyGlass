import os
from openai import OpenAI
from dotenv import load_dotenv
from pyht import Client
from pyht.client import TTSOptions, Language

# Load environment variables
load_dotenv('.env.local')

def generate_counter_message(analysis, original_messages=None, intent="diplomatic"):
    client = OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.omnistack.sh/openai/v1"),
        api_key=os.getenv("OMNISTACK_API_KEY")
    )

    # Create context from original messages
    context = "\n\nOriginal intercepted messages:\n"
    if original_messages:
        for msg in original_messages:
            context += f"\n- {msg['content']}\n"

    counter_msg_response = client.chat.completions.create(
        model="casandra_bernard_sheridan",
        messages=[
            {"role": "system", "content": """You are a counter-intelligence expert for the Ukraine army. 
            Analyze the intercepted messages and analysis to generate a counter-message.
            Your response must:
            - Only include a single short sentence to be sent to the enemy.
            - Do not include any coordinates and preferable no numbers.
            - Provide false information based on the intended counter-message.
            - Be in a way that people talk, not so much like written text.
            - Be concise and provide no explanation, analysis, or additional text"""
             },
            {"role": "user", "content": f"Analysis of enemy communications: {analysis}\n{context}"}
        ]
    )
    counter_msg_english = counter_msg_response.choices[0].message.content

    # Generate Ukrainian translation
    translation_response = client.chat.completions.create(
        model="casandra_bernard_sheridan",
        messages=[
            {"role": "system", "content": """You are an expert translator.
             Translate the message from English to Ukrainian.
             Provide only the translated message, no additional text."""
             },
            {"role": "user", "content": counter_msg_english}
        ]
    )
    counter_msg_ukrainian = translation_response.choices[0].message.content

    # Generate audio files
    client_pyth = Client(
        user_id=os.getenv("PLAYHT_USER_ID"),
        api_key=os.getenv("PLAYHT_API_KEY"),
    )

    # Generate English audio
    options = TTSOptions(
        language=Language.ENGLISH,
        voice="s3://voice-cloning-zero-shot/36e9c53d-ca4e-4815-b5ed-9732be3839b4/samuelsaad/manifest.json"
    )
    
    with open("output_english.mp3", "wb") as f:
        for chunk in client_pyth.tts(counter_msg_english, options):
            f.write(chunk)
    print("Audio saved to output_english.mp3")

    # Generate Ukrainian audio
    options = TTSOptions(
        language=Language.UKRAINIAN,
        voice="s3://voice-cloning-zero-shot/36e9c53d-ca4e-4815-b5ed-9732be3839b4/samuelsaad/manifest.json"
    )
    
    with open("output_ukrainian.mp3", "wb") as f:
        for chunk in client_pyth.tts(counter_msg_ukrainian, options):
            f.write(chunk)
    print("Audio saved to output_ukrainian.mp3")

    return counter_msg_english, counter_msg_ukrainian