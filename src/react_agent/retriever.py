import os
import requests
from typing import List
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PlaywrightURLLoader
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    WebBaseLoader,
)
import pandas as pd
from datetime import datetime


class VectorStoreManager:
    """Singleton class to manage the vector store initialization and retrieval."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorStoreManager, cls).__new__(cls)
            cls._instance._initialize_vectorstore()
        return cls._instance

    def _initialize_vectorstore(self):
        """Initialize the vector store only once."""
        print("Initializing vector store...")
        embedding = OpenAIEmbeddings()
        current_date = datetime.now().strftime("%d_%m_%Y")
        trop_url = "https://mausam.imd.gov.in/backend/assets/cyclone_pdf/Tropical_Weather_Outlook_based_on_0300_UTC_of_23_01_2025.pdf"
        modified_url = trop_url.replace("23_01_2025", current_date)

        urls = [
            "https://mausam.imd.gov.in/chennai/mcdata/fishermen.pdf",
            "https://mausam.imd.gov.in/chennai/mcdata/daily_weather_report.pdf",
            modified_url,
        ]

        documents = self._load_documents_from_urls(urls)
        cleaned_documents = [
            Document(
                page_content=self._clean_document_content(doc.page_content),
                metadata=doc.metadata,
            )
            for doc in documents
        ]
        split_documents_list = self._split_documents(cleaned_documents)

        self.vectorstore = Chroma.from_documents(
            documents=split_documents_list,
            collection_name="rag-chroma",
            embedding=embedding,
        )
        print("Vector store initialized.")

    def get_retriever(self):
        """Return the retriever from the initialized vector store."""
        return self.vectorstore.as_retriever()

    def _load_documents_from_urls(self, urls: List[str]) -> List[Document]:
        """Load documents from a list of URLs."""
        docs = []
        for url in urls:
            if url.endswith(".pdf"):
                response = requests.get(url)
                pdf_path = url.split("/")[-1]
                with open(pdf_path, "wb") as file:
                    file.write(response.content)
                loader = PyPDFLoader(pdf_path)
                docs.extend(loader.load())
            else:
                loader = (
                    PlaywrightURLLoader(url)
                    if url.startswith("https://city.imd.gov.in")
                    else WebBaseLoader(url)
                )
                docs.extend(loader.load())
        return docs

    def _clean_document_content(self, doc_content: str) -> str:
        """Clean document content by removing HTML tags and unnecessary whitespace."""
        if "<" in doc_content and ">" in doc_content:
            soup = BeautifulSoup(doc_content, "html.parser")
            doc_content = soup.get_text()
        return doc_content.replace("\n", " ").replace("\r", " ").strip()

    def _split_documents(
        self, documents: List[Document], chunk_size=500, chunk_overlap=0
    ) -> List[Document]:
        """Split documents into smaller chunks."""
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        return text_splitter.split_documents(documents)


# Ensure the vector store is initialized at import time
vector_store_manager = VectorStoreManager()

if __name__ == "__main__":
    retriever = vector_store_manager.get_retriever()
    question = "Is there any cyclone alert or any weather warning in Tamil Nadu?"
    result = retriever.invoke(question)
    print(f"Answer: {result}")
