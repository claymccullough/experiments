from dotenv import load_dotenv

load_dotenv('.env')

import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import JSONLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores.chroma import Chroma

"""
GLOBALS
"""
FILE_PATH = "./data/partial.json"
MODEL_NAME = os.environ.get("MODEL_NAME")
CHUNK_SIZE = 4000
CHUNK_OVERLAP = 200


def metadata_func(record: dict, metadata: dict) -> dict:
    return {
        'source': record.get('source'),
        'title': record.get('title'),
    }


if __name__ == '__main__':
    print('STARTING INGEST')

    print('JSONLOADER...')
    json_loader = JSONLoader(
        file_path=FILE_PATH,
        jq_schema=".[]",
        content_key="page_content",
        metadata_func=metadata_func
    )

    print('SPLITTING NEW DOCUMENTS INTO CHUNKS...')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs = text_splitter.split_documents(json_loader.load())

    # Load into chroma
    print('LOADING INTO CHROMADB...')
    embedding_function = OllamaEmbeddings(model=MODEL_NAME, num_gpu=1, num_thread=6)
    db = Chroma.from_documents(docs, embedding_function, persist_directory="./chroma_db")

    print('DONE')