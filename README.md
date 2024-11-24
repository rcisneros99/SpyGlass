# SpyGlass 🇺🇦

SpyGlass is an intelligence analysis system that processes intercepted audio communications, generates transcripts, analyzes content, and helps generate counter-intelligence messages.

## Features

- 🎙️ Audio Processing: Process audio files to generate transcripts
- 🔍 Intelligence Analysis: Analyze transcripts using advanced embedding search
- 💬 Interactive Chat: Query the intelligence database through natural language
- 🎯 Counter-Intelligence: Generate strategic counter-messages in English and Ukrainian
- 🔊 Audio Generation: Create audio versions of counter-messages with realistic effects

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/spyglass.git
cd spyglass
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env.local` file with your API keys:
```env
OPENAI_API_KEY=your_openai_key
PLAYHT_USER_ID=your_playht_user_id
PLAYHT_API_KEY=your_playht_api_key
OMNISTACK_API_KEY=your_omnistack_key
OPENAI_BASE_URL=https://api.omnistack.sh/openai/v1
```

## Project Structure

```
spyglass/
├── data/
│   ├── uploads/      # Place audio files here
│   └── transcripts/  # Generated transcripts and embeddings
├── scripts/
│   ├── app.py        # Streamlit web interface
│   ├── ben.py        # Audio processing module
│   ├── dexter_doc_embeddings.py  # Document embedding module
│   └── rafa.py       # Counter-message generation module
├── .env.local        # Environment variables
└── requirements.txt  # Python dependencies
```

## Usage

1. Place audio files in the `data/uploads` directory

2. Run the Streamlit app:
```bash
streamlit run scripts/app.py
```

3. Use the web interface to:
   - Upload new audio files
   - View critical intelligence
   - Query the intelligence database
   - Generate counter-messages
   - Listen to generated audio responses

## Dependencies

- Python 3.8+
- OpenAI API access
- PlayHT API access
- Streamlit
- Other requirements listed in requirements.txt

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 