import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()


def main(): ...


if __name__ == "__main__":
    loader = TextLoader(
        r"C:\Users\hfdkw\OneDrive\Рабочий стол\langchain_learn\langchain-course\rag\mediumblog1.txt",
        encoding="UTF-8",
    )
    document = loader.load()
    print("splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents=document)
    print(f"Created: {len(texts)} chunks")
    embeddings = GoogleGenerativeAIEmbeddings(
        google_api_key=os.environ.get("GOOGLE_API_KEY", ""),
        model="gemini-embedding-2-preview",
        output_dimensionality=1536,
    )
    PineconeVectorStore.from_documents(
        texts, embeddings, index_name=os.environ.get("INDEX_NAME", "")
    )
    print("Ingested")
