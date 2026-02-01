# 🌸 Chiy0: Hybrid Fine-Tuned + RAG LLM Assistant

This repository contains **Chiy0**, a hybrid LLM assistant that combines **Retrieval-Augmented Generation (RAG)** with **LoRA fine-tuning via Unsloth**.

The goal of this project is to explore a **full local knowledge pipeline** from automated document acquisition, to embedding-based retrieval, to model specialization via knowledge distillation while keeping inference fast and modular.

The project is named after **Chiyo-chan from one of my favorite animes, *Azumanga Daioh***.

---

##  Core Ideas

Chiy0 is built around three core ideas:

1. **Knowledge should be grounded** (RAG)
2. **Models should specialize cheaply** (LoRA + distillation)
3. **Data should be collected automatically** (PDF scraping)

---

##  Document Acquisition

Chiy0 uses the **Serper API** to search Google for relevant PDF documents based on a user’s high-level intent.  
An LLM converts vague user input (e.g. *“semiconductor industry”*) into a concise search query, which is then used to download PDFs automatically.

These documents form the shared knowledge base for both RAG and fine-tuning.

---

##  Document Processing & RAG

Downloaded PDFs are:

1. Parsed and chunked
2. Embedded using **Ollama (`nomic-embed-text`)**
3. Stored in a **Chroma vector database**

During inference, user queries retrieve the most relevant chunks, which are injected into the model prompt to ground responses in source documents.

---

##  Knowledge Distillation & Fine-Tuning

Chiy0 includes a **teacher–student distillation pipeline**:

- A larger **teacher model** generates and evaluates Q&A pairs from document chunks
- High-quality samples are filtered automatically
- A smaller **student model** is fine-tuned using **LoRA** via Unsloth

This allows rapid domain adaptation without full model retraining.

---

## Hybrid Inference

At runtime, Chiy0 performs **hybrid inference**:

- Retrieval from the vector database (RAG)
- Generation using a **LoRA-fine-tuned local model**
- Strict prompting to prevent hallucination when context is missing


##  Implementation Notes

- RAG and fine-tuning share the **same document source**
- Chunk IDs are deterministic to avoid duplicate embeddings
- Teacher self-evaluation filters low-quality synthetic data
- LoRA keeps training lightweight and reversible
- Prompts explicitly forbid reasoning leakage during inference

---

##  Disclaimer

This project is for **research and educational purposes**.  
Downloaded documents remain the property of their respective owners.

---

