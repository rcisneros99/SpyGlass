from pyht import Client
from pyht.client import TTSOptions, Language

client = Client(
    user_id='2tI6DaaykKWdxKGVh4U91Wh9ZCx2',
    api_key='7f2d34f49ae84bbb92d4752e38769e53',
)

options = TTSOptions(
    language=Language.UKRAINIAN,
    voice="s3://voice-cloning-zero-shot/36e9c53d-ca4e-4815-b5ed-9732be3839b4/samuelsaad/manifest.json"
)

# Generate speech for the input text
try:
    with open("output_ukrainian.mp3", "wb") as f:
        for chunk in client.tts("Ми зараз їдемо в місто", options):
            f.write(chunk)
    print("Audio saved to output_ukrainian.mp3")
except Exception as e:
    print(f"An error occurred: {e}")