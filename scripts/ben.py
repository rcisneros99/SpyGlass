import os
import logging
from datetime import datetime
import pandas as pd
from openai import OpenAI
import asyncio

class AudioProcessor:
    def __init__(self, input_dir: str, output_dir: str, openai_api_key: str):
        self.input_dir = os.path.abspath(os.path.expanduser(input_dir))
        self.output_dir = os.path.abspath(os.path.expanduser(output_dir))
        self.openai_client = OpenAI(
            api_key=openai_api_key,
            base_url="https://api.openai.com/v1"
        )
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        os.makedirs(self.output_dir, exist_ok=True)

    def get_audio_files(self):
        supported_formats = ('.mp3', '.wav', '.m4a', '.opus', '.flac', '.aac')
        audio_files = []
        
        for root, _, filenames in os.walk(self.input_dir):
            for filename in filenames:
                if filename.lower().endswith(supported_formats):
                    full_path = os.path.join(root, filename)
                    audio_files.append(full_path)
        
        return audio_files

    async def translate_file(self, file_path: str) -> str:
        try:
            with open(file_path, "rb") as audio_file:
                translation = self.openai_client.audio.translations.create(
                    model="whisper-1",
                    file=audio_file
                )
            return translation.text
                
        except Exception as e:
            self.logger.error(f"Error translating {file_path}: {str(e)}")
            raise e

    async def process_single_file(self, file_path: str) -> dict:
        """Process a single file through translation."""
        try:
            # Translate
            translation = await self.translate_file(file_path)
            
            # Save translation
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_file = os.path.join(self.output_dir, f"{base_name}_translation.txt")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(translation)
            
            return {
                'filename': os.path.basename(file_path),
                'translation': translation,
                'timestamp': datetime.now().isoformat(),
                'status': 'success'
            }
            
        except Exception as e:
            return {
                'filename': os.path.basename(file_path),
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'status': 'error'
            }

    async def process_files(self):
        """Process all audio files."""
        audio_files = self.get_audio_files()
        self.logger.info(f"Found {len(audio_files)} audio files to process")
        
        all_results = []
        for audio_file in audio_files:
            result = await self.process_single_file(audio_file)
            all_results.append(result)
            
            if result['status'] == 'success':
                self.logger.info(f"✓ Processed {result['filename']}")
            else:
                self.logger.error(f"✗ Failed {result['filename']}: {result.get('error')}")
        
        # Save summary report
        df = pd.DataFrame(all_results)
        report_path = os.path.join(self.output_dir, 'translation_report.csv')
        df.to_csv(report_path, index=False)
        self.logger.info(f"Translation report saved to {report_path}")
        
        return all_results