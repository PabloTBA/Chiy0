import torch
import json
import re
import os
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from transformers import TextStreamer

class KnowledgeDistiller:
    def __init__(self, teacher_id: str, student_id: str, output_path: str = "synthetic_data.json"):
        self.teacher_id = teacher_id
        self.student_id = student_id
        self.output_path = output_path
        
        # Tools
        self.converter = DocumentConverter()
        self.chunker = HybridChunker()
        
        # Model placeholders
        self.model = None
        self.tokenizer = None

    ## --- PHASE 1: DOCUMENT INGESTION ---
    
    def process_pdfs(self, folder_path: str) -> List[Any]:
        """Converts and chunks PDFs from a directory."""
        pdf_files = list(Path(folder_path).glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in {folder_path}")

        all_chunks = []
        print(f"Found {len(pdf_files)} PDFs. Processing...")
        
        for pdf_path in tqdm(pdf_files, desc="Ingesting PDFs"):
            try:
                doc = self.converter.convert(pdf_path)
                chunks = self.chunker.chunk(dl_doc=doc)
                # Contextualize each chunk
                file_chunks = [self.chunker.contextualize(c) for c in chunks]
                all_chunks.extend(file_chunks)
            except Exception as e:
                print(f"Error processing {pdf_path.name}: {e}")
        
        return all_chunks

    ## --- PHASE 2: TEACHER GENERATION ---

    def load_teacher(self):
        """Loads the teacher model for generation/judging."""
        print(f"\n--- Loading Teacher: {self.teacher_id} ---")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.teacher_id,
            max_seq_length=4096,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(self.model)

    def _query_teacher(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors='pt'
        ).to("cuda")

        outputs = self.model.generate(
            input_ids=inputs,
            max_new_tokens=512,
            use_cache=True,
            temperature=0.7
        )
        return self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)

    def generate_synthetic_data(self, chunks: List[Any], min_score: int = 4) -> List[Dict]:
        """Generates and filters Q&A pairs using the Teacher model."""
        dataset = []
        
        for i, chunk in enumerate(chunks):
            chunk_text = getattr(chunk, 'text', str(chunk)) # Ensure we get string
            print(f"Processing chunk {i+1}/{len(chunks)}...")

            # 1. Generate Q&A
            prompt = f"Analyze the text and generate 3 high-quality Q&A pairs in JSON format: {chunk_text}"
            response = self._query_teacher(prompt)
            
            try:
                match = re.search(r'\[.*\]', response, re.DOTALL)
                qa_pairs = json.loads(match.group(0)) if match else []
            except:
                continue

            # 2. Judge Q&A
            for pair in qa_pairs:
                judge_prompt = f"Rate this Q&A pair 1-5 on accuracy: Q: {pair['question']} A: {pair['answer']}. Output ONLY the digit."
                score_resp = self._query_teacher(judge_prompt)
                
                try:
                    score = int(re.search(r'\d', score_resp).group(0))
                except:
                    score = 3

                if score >= min_score:
                    dataset.append(pair)
                    print(f"  [ACCEPTED] Score {score}")

        # Save results
        with open(self.output_path, "w") as f:
            json.dump(dataset, f, indent=2)
            
        return dataset

    ## --- PHASE 3: STUDENT TRAINING ---

    def train_student(self, qa_data: List[Dict]):
        """Fine-tunes the student model using the generated dataset."""
        print(f"\n--- Training Student: {self.student_id} ---")
        
        # Cleanup Teacher from VRAM
        del self.model, self.tokenizer
        torch.cuda.empty_cache()

        # Prepare Dataset
        hf_dataset = Dataset.from_list(qa_data)
        def format_fn(batch):
            texts = [f"<|start_header_id|>user<|end_header_id|>\n\n{q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{a}<|eot_id|>" 
                     for q, a in zip(batch["question"], batch["answer"])]
            return {"text": texts}
        
        hf_dataset = hf_dataset.map(format_fn, batched=True)

        # Load Student
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.student_id,
            max_seq_length=2048,
            load_in_4bit=True,
        )

        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
        )

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=hf_dataset,
            dataset_text_field="text",
            max_seq_length=2048,
            args=SFTConfig(
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                num_train_epochs=1,
                learning_rate=2e-4,
                logging_steps=1,
                optim="adamw_8bit",
                output_dir="outputs",
            ),
        )
        
        trainer.train()
        self.model.save_pretrained("lora_model")
        self.tokenizer.save_pretrained("lora_model")
        print("Training Complete & Saved!")

# --- EXECUTION ---

#if __name__ == "__main__":
    #distiller = KnowledgeDistiller(
        #teacher_id='unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit',
        #student_id='unsloth/Llama-3.2-3B-Instruct'
    #)

    # Step 1: Ingest
    #chunks = distiller.process_pdfs('finetune_pdfs/')
    
    # Step 2: Generate
    #distiller.load_teacher()
    #qa_data = distiller.generate_synthetic_data(chunks)
    
    # Step 3: Train
    #if qa_data:
        #distiller.train_student(qa_data)