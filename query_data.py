import torch
from unsloth import FastLanguageModel
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from embedding_function import get_embedding_function

class RAGQueryEngine:
    def __init__(self, chroma_path="chroma", lora_path="./lora_model"):
        self.chroma_path = chroma_path
        
        # 1. Initialize DB
        self.embedding_function = get_embedding_function()
        self.db = Chroma(
            persist_directory=self.chroma_path, 
            embedding_function=self.embedding_function
        )
        
        # 2. Load Model & Tokenizer
        print(f"Loading Unsloth optimized model from {lora_path}...")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = lora_path,
            max_seq_length = 2048,
            load_in_4bit = True,
            device_map = "auto",
        )
        FastLanguageModel.for_inference(self.model)

    def query(self, query_text: str):
        # 1. Search the DB
        results = self.db.similarity_search_with_score(query_text, k=5)
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results]) if results else "No context."

        # 2. Format with Llama-3 Chat Template (Crucial for preventing loops)
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer the question briefly and directly. If the answer is not in the context, say you don't know. Do not provide notes, hints, or explanations about your thought process."},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query_text}"}
        ]
        
        # Apply the official chat template
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True, # This adds the <|start_header_id|>assistant tag
            return_tensors="pt",
        ).to("cuda")

        # 3. Generate with Repetition Penalty
        outputs = self.model.generate(
            input_ids=inputs,
            max_new_tokens=256,
            temperature=0.1,
            repetition_penalty=1.2, # Stops the "Note: Note: Note:" loops
            use_cache=True,
            eos_token_id=self.tokenizer.eos_token_id
        )

        # 4. Decode ONLY the new tokens (the answer)
        # We slice the output to remove the input prompt tokens
        generated_tokens = outputs[0][len(inputs[0]):]
        response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return {
            "answer": response_text.strip(),
            # Sources are returned but can be ignored in your main.py display
            "sources": [doc.metadata.get("id", "Unknown") for doc, _score in results]
        }