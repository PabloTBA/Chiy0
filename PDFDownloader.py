import os
import requests
import json
import re
import time
from dotenv import load_dotenv
from transformers import TextStreamer
from unsloth import FastLanguageModel

class Downloader:
    def __init__(self):
        # 1. Load Credentials on Initialization
        load_dotenv()
        self.SERPER_API_KEY = os.getenv("SERPER_API_KEY")
        if not self.SERPER_API_KEY:
            raise ValueError(" Error: SERPER_API_KEY not found in .env file.")
        
        # Define headers once since they are reused
        self.search_headers = {
            'X-API-KEY': self.SERPER_API_KEY,
            'Content-Type': 'application/json'
        }
        self.dl_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

    def sanitize_filename(self, title):
        # Fix: Use the 'title' argument, not self.title
        return re.sub(r'[\\/*?:"<>|]', "_", title).strip()[:60]

    def serper_search_and_download(self, query, num_files=3, download_folder="finetune_pdfs"):
        # Fix: Use local variable 'download_folder', not self.download_folder
        if not os.path.exists(download_folder):
            os.makedirs(download_folder)

        print(f" Serper Search for: '{query}'...")
        
        url = "https://google.serper.dev/search"
        payload = json.dumps({
            "q": f"{query} filetype:pdf",
            "num": 20 
        })

        try:
            # Fix: Use 'url' and 'payload', no need to store them in 'self'
            response = requests.post(url, headers=self.search_headers, data=payload)
            response.raise_for_status()
            results = response.json().get('organic', [])
        except Exception as e:
            print(f" API Error: {e}")
            return

        if not results:
            print(" No results found.")
            return

        download_count = 0
        
        for item in results:
            if download_count >= num_files:
                break

            pdf_url = item.get('link')
            title = item.get('title', 'document')
            
            # Check for PDF extension
            if not pdf_url.lower().endswith('.pdf'):
                continue
            
            # Fix: Call self.sanitize_filename
            safe_filename = self.sanitize_filename(title)
            save_path = os.path.join(download_folder, f"{safe_filename}.pdf")

            try:
                print(f" Downloading: {safe_filename}...")
                file_response = requests.get(pdf_url, headers=self.dl_headers, timeout=15)
                file_response.raise_for_status()
                
                with open(save_path, 'wb') as f:
                    f.write(file_response.content)
                
                print(f" Saved to: {save_path}")
                download_count += 1
                time.sleep(1)
                
            except Exception as e:
                print(f" Failed to download {pdf_url}: {e}")

        print(f"\n Task Complete. Downloaded {download_count}/{num_files} files.")
    
    def getUserNeeds(self, num_files=2):
        from unsloth import FastLanguageModel
        import torch # Needed for decoding
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = 'unsloth/Llama-3.2-3B-Instruct',
            max_seq_length = 4096,
            load_in_4bit = True,
            dtype = None,
        )
        FastLanguageModel.for_inference(model)
        
        user_input = input("What industry or company type are you interested in? ")
        
        # We need to tell the LLM to provide ONLY a search query
        prompt = (
            "Based on the user's interest, provide a concise, 3-5 word search query "
            "to find relevant PDF documents (manuals, reports, or guides) for fine-tuning. "
            "Output ONLY the search query text.\n\n"
            f"User Interest: {user_input}"
        )
        
        messages = [{"role": "user", "content": prompt}]
        
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize = True,
            add_generation_prompt = True,
            return_tensors = "pt"
        ).to("cuda")

        # Generate output
        outputs = model.generate(
            input_ids = inputs, 
            max_new_tokens = 64,
            use_cache = True
        )
        
        # FIX: Extract only the NEW tokens and decode to a string
        # This removes the prompt and gives you just the LLM's answer
        generated_ids = [output[len(inputs[0]):] for output in outputs]
        decoded_output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        print(f"\n--- LLM Generated Search Query: {decoded_output} ---\n")
        
        # Pass the clean string to the search function
        self.serper_search_and_download(decoded_output, num_files)

    def test(self):
    
        base_llm_output = "Quant Finance Basics"
        self.serper_search_and_download(base_llm_output, num_files=5)

# --- EXECUTION ---
if __name__ == "__main__":
    downloader = Downloader()
    downloader.test()