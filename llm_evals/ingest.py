# import pysqlite3
# import sys
# sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

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
FILE_PATH = "./data/cities.json"
MODEL_NAME = os.environ.get("EMBED_MODEL_NAME")
EMBED_BASE_URL = os.environ.get("EMBED_BASE_URL")
CHUNK_SIZE = 4000
CHUNK_OVERLAP = 200
DEST_PATH = os.environ.get("DEST_PATH")


def metadata_func(record: dict, metadata: dict) -> dict:
    return {
        'source': record.get('source'),
        'title': record.get('title'),
        'file_name': './data/cities.json'
    }


def get_embeddings():
    return OllamaEmbeddings(model=MODEL_NAME, base_url=EMBED_BASE_URL)

def get_docs():
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
    return docs



if __name__ == '__main__':
    print('STARTING INGEST')

    docs = get_docs()

    # Load into chroma
    print(f'LOADING {len(docs)} DOCS INTO CHROMADB...')
    embedding_function = get_embeddings()
    db = Chroma.from_documents(docs, embedding_function, persist_directory=DEST_PATH)

    print('DONE')
