import unsloth # MUST BE FIRST
import os
import sys
import torch
from PDFDownloader import Downloader
from handle_database import DocumentManager
from Finetuner import KnowledgeDistiller
from query_data import RAGQueryEngine
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
def print_header(title):
    print(f"\n{'='*50}")
    print(f"   {title}")
    print(f"{'='*50}")

def main():
    downloader = Downloader()
    db_manager = DocumentManager(data_path="finetune_pdfs", chroma_path="chroma")
    
    rag_engine = None

    while True:
        print("\n---  LOCAL AI HUB MENU ---")
        print("1. [Data]  Download PDFs (Industry-based Search)")
        print("2. [RAG]   Index Downloaded PDFs to Vector DB")
        print("3. [RAG]   Clear Vector Database (Reset)")
        print("4. [Train] Run Knowledge Distillation & Fine-tuning")
        print("5. [Chat]  Query your Fine-tuned Model (with RAG)")
        print("0. Exit")
        
        choice = input("\nSelect an option: ")

        if choice == '1':
            print_header("PDF DOWNLOADER")
            num = input("How many files to download? (Default 2): ")
            num_files = int(num) if num.strip() else 2
            downloader.getUserNeeds(num_files=num_files)

        elif choice == '2':
            print_header("DATABASE INDEXING")
            print("Processing PDFs and updating ChromaDB...")
            db_manager.run_indexing_pipeline()
            print("Successfully indexed documents.")

        elif choice == '3':
            confirm = input(" Are you sure you want to delete the database? (y/n): ")
            if confirm.lower() == 'y':
                db_manager.clear_database()

        elif choice == '4':
            print_header("FINE-TUNING PIPELINE")
            # Clear memory before training
            if rag_engine:
                del rag_engine
                torch.cuda.empty_cache()
            
            distiller = KnowledgeDistiller(
                teacher_id='unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit',
                student_id='unsloth/Llama-3.2-3B-Instruct'
            )
            
            print("Phase 1: Ingesting PDFs...")
            chunks = distiller.process_pdfs('finetune_pdfs/')
            
            print("Phase 2: Teacher Generating Synthetic Data...")
            distiller.load_teacher()
            qa_data = distiller.generate_synthetic_data(chunks)
            
            if qa_data:
                print("Phase 3: Fine-tuning Student (LoRA)...")
                distiller.train_student(qa_data)
            else:
                print("Error: No synthetic data was generated.")

        elif choice == '5':
            print_header("RAG CHAT (FINE-TUNED MODEL)")
            
            if not os.path.exists("./lora_model"):
                print(" Error: No fine-tuned model found. Please run Option 4 first.")
                continue

            if rag_engine is None:
                print("Initial loading of LoRA model via Unsloth...")
                rag_engine = RAGQueryEngine(chroma_path="chroma", lora_path="./lora_model")

            while True:
                user_q = input("\nQuestion (type 'back' to return): ")
                if user_q.lower() == 'back':
                    break
                
                try:
                    result = rag_engine.query(user_q)
                    print(f"\n AI RESPONSE:\n{result['answer']}")
                    print(f"\n SOURCES: {result['sources']}")
                except Exception as e:
                    print(f"Error during query: {e}")

        elif choice == '0':
            print("Exiting Hub. Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    os.makedirs("finetune_pdfs", exist_ok=True)
    os.makedirs("chroma", exist_ok=True)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\nShutdown requested by user.")
        sys.exit(0)