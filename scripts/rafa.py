from openai import OpenAI
from pyht import Client
from pyht.client import TTSOptions, Language
from dotenv import load_dotenv
import os
import soundfile
import numpy as np

# Load environment variables
load_dotenv('.env.local')

def generate_counter_message(analysis, original_messages=None, intent="diplomatic"):
    client = OpenAI(
        base_url="https://api.omnistack.sh/openai/v1",
        api_key=os.getenv("OMNISTACK_API_KEY")
    )

    # Create message content from analysis and original messages
    message_content = f"""Enemy message: {analysis}
Counter-message intention: {intent}"""
    if original_messages:
        message_content += "\nAdditional context:\n"
        for msg in original_messages:
            message_content += f"- {msg['content']}\n"

    # Generate counter message
    counter_msg_generated = client.chat.completions.create(
        model="casandra_bernard_sheridan",
        messages=[
            {"role": "system", "content": """You are a counter-intelligence expert for the Ukraine army. 
            Analyze the intercepted message and generate a counter-message based on the provided intention. 

            Your response must:
            - Only include a single short sentence to be sent to the enemy.
            - Do not include any coordinates and preferable no numbers.
            - Provide false information that aligns with the counter-message intention.
            - Be in a way that people talk, not so much like written text.
            - Be concise and provide no explanation, analysis, or additional text"""
             },
            {"role": "user", "content": message_content}
        ]
    )
    counter_msg_english = counter_msg_generated[0].choices[0].message.content

    # Generate Ukrainian translation
    translated_counter_msg = client.chat.completions.create(
        model="casandra_bernard_sheridan",
        messages=[
            {"role": "system", "content": """You are an expert translator and you know exactly how people communicate in English and Ukranian.
             Your job is to translate the message from English to Ukrainian.
             Your response must have:
             - The translated message
             - No additional text, analysis, or explanation
             - It must be in a way that people would actually say it in real life, include pauses, filler words, etc. not like a formal letter or something.
             """
             },
            {"role": "user", "content": counter_msg_english}
        ]
    )
    translated_counter_msg = translated_counter_msg[0].choices[0].message.content

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

    try:
        with open("output_english.mp3", "wb") as f:
            for chunk in client_pyth.tts(counter_msg_english, options):
                f.write(chunk)
        print("Audio saved to output_english.mp3")
    except Exception as e:
        print(f"An error occurred generating English audio: {e}")

    # Generate Ukrainian audio
    options = TTSOptions(
        language=Language.UKRAINIAN,
        voice="s3://voice-cloning-zero-shot/36e9c53d-ca4e-4815-b5ed-9732be3839b4/samuelsaad/manifest.json"
    )

    try:
        # First generate clean audio
        clean_audio_chunks = []
        for chunk in client_pyth.tts(translated_counter_msg, options):
            clean_audio_chunks.append(chunk)
        
        with open("output_ukrainian_clean.mp3", "wb") as f:
            for chunk in clean_audio_chunks:
                f.write(chunk)
        # Read the clean audio and add noise
        audio_data, sample_rate = soundfile.read("output_ukrainian_clean.mp3")
        
        # Generate stronger noise
        noise = np.random.normal(0, 0.05, len(audio_data))  # Increased noise amplitude
        
        # Add some static/crackle effects
        static = np.random.uniform(-0.02, 0.02, len(audio_data))
        
        # Add some frequency modulation
        t = np.linspace(0, len(audio_data)/sample_rate, len(audio_data))
        modulation = 0.02 * np.sin(2 * np.pi * 2 * t)  # 2 Hz modulation
        
        # Combine all effects
        noisy_audio = audio_data + noise + static + modulation
        
        # Add some random dropouts
        dropout_mask = np.random.random(len(audio_data)) > 0.01  # 1% chance of dropout
        noisy_audio = noisy_audio * dropout_mask
        
        # Clip to prevent distortion
        noisy_audio = np.clip(noisy_audio, -1, 1)
        
        # Save the noisy audio
        soundfile.write('output_ukrainian.mp3', noisy_audio, sample_rate)
        print("Audio saved to output_ukrainian.mp3")


    except Exception as e:
        print(f"An error occurred generating Ukrainian audio: {e}")

    return counter_msg_english, translated_counter_msg

if __name__ == "__main__":
    # Test the function
    test_msg = "The location of the next attack is 50.4504° N, 30.5245° E"
    english, ukrainian = generate_counter_message(test_msg)
    print(f"English: {english}")
    print(f"Ukrainian: {ukrainian}")