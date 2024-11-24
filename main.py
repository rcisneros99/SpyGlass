from openai import OpenAI


crit_msg = 'The location of the next attack is 50.4504° N, 30.5245° E'

counter_msg = 'Stay in the city'

client = OpenAI(
    base_url="https://api.omnistack.sh/openai/v1",
    api_key="osk_5b0927be8cb994a3670d7468f661c7db"
)

counter_msg_generated = client.chat.completions.create(
    model="casandra_bernard_sheridan",
    messages=[
        {"role": "system", "content": """You are a counter-intelligence expert for the Ukraine army. 
        Analyze the intercepted message and the intended counter-message. 

        Your response must:
        - Only include a single short sentence to be sent to the enemy.
        - Do not include any coordinates and preferable no numbers.
        - Provide false information based on the intended counter-message.
        - Be in a way that people talk, not so much like written text.
        - Be concise and provide no explanation, analysis, or additional text"""
         },
        {"role": "user", "content": "Enemy message: " + crit_msg + ". Counter-message intention: " + counter_msg}
    ]
)
counter_msg_english = counter_msg_generated[0].choices[0].message.content

print(counter_msg_english)

translated_counter_msg = client.chat.completions.create(
    model="casandra_bernard_sheridan",
    messages=[
        {"role": "system", "content": """You are an expert translator and you know exactly how people communicate in English and Ukranian.
         Your job is to translate the message from English to Ukrainian.
         Your response must have:
         - The translated message
         - No additional text, analysis, or explanation
         - It must be in a way that people would actually say it in real life, not like a formal letter or something.
         """
         },
        {"role": "user", "content": "Enemy message: " + crit_msg + ". Counter-message intention: " + counter_msg}
    ]
)
translated_counter_msg = translated_counter_msg[0].choices[0].message.content

print(translated_counter_msg)

from pyht import Client
from pyht.client import TTSOptions, Language

client_pyth = Client(
    user_id='2tI6DaaykKWdxKGVh4U91Wh9ZCx2',
    api_key='7f2d34f49ae84bbb92d4752e38769e53',
)

options = TTSOptions(
    language=Language.ENGLISH,
    voice="s3://voice-cloning-zero-shot/36e9c53d-ca4e-4815-b5ed-9732be3839b4/samuelsaad/manifest.json"
)

# Generate speech for the input text
try:
    with open("output_english.mp3", "wb") as f:
        for chunk in client_pyth.tts(counter_msg_english, options):
            f.write(chunk)
    print("Audio saved to output_english.mp3")
except Exception as e:
    print(f"An error occurred: {e}")



options = TTSOptions(
    language=Language.UKRAINIAN,
    voice="s3://voice-cloning-zero-shot/36e9c53d-ca4e-4815-b5ed-9732be3839b4/samuelsaad/manifest.json"
)

# Generate speech for the input text
try:
    with open("output_ukrainian.mp3", "wb") as f:
        for chunk in client_pyth.tts(translated_counter_msg, options):
            f.write(chunk)
    print("Audio saved to output_ukrainian.mp3")
except Exception as e:
    print(f"An error occurred: {e}")