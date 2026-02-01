import argparse
import os
import shutil
from langchain_text_splitters import RecursiveCharacterTextSplitter
from embedding_function import get_embedding_function
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document

class DocumentManager:
    def __init__(self, data_path: str = "company_docs", chroma_path: str = "chroma"):
        self.data_path = data_path
        self.chroma_path = chroma_path
        self.embedding_function = get_embedding_function()

    def load_documents(self) -> list[Document]:
        """Loads PDFs from the specified directory."""
        loader = PyPDFDirectoryLoader(self.data_path)
        return loader.load()

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """Splits documents into smaller chunks."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=80,
            length_function=len,
            is_separator_regex=False,
        )
        return text_splitter.split_documents(documents)

    def calculate_chunk_ids(self, chunks: list[Document]) -> list[Document]:
        """Creates unique IDs: 'source:page:chunk_index'."""
        last_page_id = None
        current_chunk_index = 0

        for chunk in chunks:
            source = chunk.metadata.get("source")
            page = chunk.metadata.get("page")
            current_page_id = f"{source}:{page}"

            if current_page_id == last_page_id:
                current_chunk_index += 1
            else:
                current_chunk_index = 0

            chunk_id = f"{current_page_id}:{current_chunk_index}"
            last_page_id = current_page_id
            chunk.metadata["id"] = chunk_id

        return chunks

    def add_to_chroma(self, chunks: list[Document]):
        """Adds unique chunks to the Chroma vector database."""
        db = Chroma(
            persist_directory=self.chroma_path, 
            embedding_function=self.embedding_function
        )

        chunks_with_ids = self.calculate_chunk_ids(chunks)
        existing_items = db.get(include=[])
        existing_ids = set(existing_items["ids"])
        
        print(f"Number of existing documents in DB: {len(existing_ids)}")

        new_chunks = [
            chunk for chunk in chunks_with_ids 
            if chunk.metadata["id"] not in existing_ids
        ]

        if new_chunks:
            print(f" Adding new documents: {len(new_chunks)}")
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            db.add_documents(new_chunks, ids=new_chunk_ids)
            # db.persist()  # Note: Recent Chroma versions persist automatically
        else:
            print(" No new documents to add")

    def clear_database(self):
        """Removes the Chroma database directory."""
        if os.path.exists(self.chroma_path):
            shutil.rmtree(self.chroma_path)
            print(" Database cleared.")

    def run_indexing_pipeline(self):
        """Orchestrates the full loading, splitting, and saving process."""
        documents = self.load_documents()
        chunks = self.split_documents(documents)
        self.add_to_chroma(chunks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()

    # Initialize the manager
    manager = DocumentManager()

    if args.reset:
        manager.clear_database()

    manager.run_indexing_pipeline()


if __name__ == "__main__":
    main()