import os
import logging
from datetime import datetime
import sieve
import pandas as pd
from openai import OpenAI
import asyncio
from tqdm import tqdm
import multiprocessing as mp
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor

class AudioProcessor:
    def __init__(self, input_dir: str, output_dir: str, openai_api_key: str):
        self.input_dir = os.path.abspath(os.path.expanduser(input_dir))
        self.output_dir = os.path.abspath(os.path.expanduser(output_dir))
        self.denoised_dir = os.path.join(self.output_dir, 'denoised')
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.num_processes = max(1, mp.cpu_count() - 1)
        self.batch_size = 10  # Number of files to process in parallel
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.denoised_dir, exist_ok=True)

    def get_audio_files(self) -> List[str]:
        supported_formats = ('.mp3', '.wav', '.m4a', '.opus', '.flac', '.aac')
        audio_files = []
        
        for root, _, filenames in os.walk(self.input_dir):
            for filename in filenames:
                if filename.lower().endswith(supported_formats):
                    full_path = os.path.join(root, filename)
                    audio_files.append(full_path)
        
        return audio_files

    async def denoise_file(self, file_path: str) -> str:
        try:
            file = sieve.File(path=file_path)
            audio_enhance = sieve.function.get("sieve/audio-enhance")
            output = audio_enhance.run(
                file,
                backend="aicoustics",
                task="all",
                enhancement_steps=64
            ).path
            
            denoised_path = os.path.join(
                self.denoised_dir, 
                os.path.basename(output)
            )
            os.replace(output, denoised_path)
            
            return denoised_path
            
        except Exception as e:
            self.logger.error(f"Error denoising {file_path}: {str(e)}")
            raise e

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

    async def process_batch(self, batch: List[str]) -> List[Dict]:
        """Process a batch of files concurrently."""
        results = []
        tasks = []
        
        # Create tasks for all files in the batch
        for file_path in batch:
            tasks.append(self.process_single_file(file_path))
        
        # Execute all tasks concurrently
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        results.extend(batch_results)
        
        return [r for r in results if r is not None]

    async def process_single_file(self, file_path: str) -> Dict:
        """Process a single file through denoising and translation."""
        try:
            # Denoise
            denoised_path = await self.denoise_file(file_path)
            
            # Translate
            translation = await self.translate_file(denoised_path)
            
            # Save translation
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_file = os.path.join(self.output_dir, f"{base_name}_translation.txt")
            with open(output_file, 'w', encoding='utf-8') as f:
                # f.write(f"Original file: {os.path.basename(file_path)}\n")
                # f.write("=" * 50 + "\n\n")
                f.write(translation)
            
            return {
                'filename': os.path.basename(file_path),
                'denoised_path': denoised_path,
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
        """Process all audio files in batches."""
        audio_files = self.get_audio_files()
        self.logger.info(f"Found {len(audio_files)} audio files to process")
        
        # Split files into batches
        batches = [audio_files[i:i + self.batch_size] 
                  for i in range(0, len(audio_files), self.batch_size)]
        
        all_results = []
        with tqdm(total=len(audio_files), desc="Processing files") as pbar:
            for batch in batches:
                results = await self.process_batch(batch)
                all_results.extend(results)
                pbar.update(len(batch))
        
        # Generate summary statistics
        successful = len([r for r in all_results if r['status'] == 'success'])
        failed = len(all_results) - successful
        
        print(f"""
        Processing completed:
        - Total files: {len(all_results)}
        - Successful: {successful}
        - Failed: {failed}
        """)
        
        # Save summary report
        df = pd.DataFrame(all_results)
        report_path = os.path.join(self.output_dir, 'translation_report.csv')
        df.to_csv(report_path, index=False)
        self.logger.info(f"Translation report saved to {report_path}")
        
        return all_results

async def main():
    INPUT_DIR = "/Users/bensunshine/Downloads/voice-analog"
    OUTPUT_DIR = "/Users/bensunshine/repos/transcriptions_full"
    OPENAI_API_KEY = "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"
    
    try:
        processor = AudioProcessor(
            input_dir=INPUT_DIR,
            output_dir=OUTPUT_DIR,
            openai_api_key=OPENAI_API_KEY
        )
        
        results = await processor.process_files()
        
        print("\nDetailed Results:")
        for result in results:
            if result['status'] == 'success':
                print(f"✓ {result['filename']}")
            else:
                print(f"✗ {result['filename']}: {result['error']}")
            
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")

if __name__ == "__main__":
    asyncio.run(main())