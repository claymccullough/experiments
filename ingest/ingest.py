# import pysqlite3
# import sys
# sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv('.env')

import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import JSONLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores.chroma import Chroma

"""
GLOBALS
"""
INGEST_BASE_PATH = "./embeddings"
INGEST_FILE_PATH = os.environ.get("INGEST_FILE_PATH")
MODEL_NAME = os.environ.get("EMBED_MODEL_NAME")
BASE_URL = os.environ.get("EMBED_BASE_URL")
INGEST_NAME = os.environ.get("INGEST_NAME")
CHUNK_SIZES = [512, 1024, 2048, 4096]
CHUNK_OVERLAPS = [0, 50, 100]
CHUNK_VARIATIONS = [(chunk_size, chunk_overlap) for chunk_size in CHUNK_SIZES for chunk_overlap in CHUNK_OVERLAPS]
CHUNK_VARIATION_FILES = [f'{INGEST_BASE_PATH}/{INGEST_NAME}x{chunk_size}x{chunk_overlap}' for chunk_size, chunk_overlap in
                         CHUNK_VARIATIONS]

print(MODEL_NAME, BASE_URL)


def metadata_func(record: dict, metadata: dict) -> dict:
    return {
        'source': record.get('source'),
        'title': record.get('title'),
    }


if __name__ == '__main__':
    print('STARTING INGEST')

    print('JSONLOADER...')
    json_loader = JSONLoader(
        file_path=INGEST_FILE_PATH,
        jq_schema=".[]",
        content_key="page_content",
        metadata_func=metadata_func
    )

    print(f'CREATING EMBEDDINGS, CHUNK_SIZES: {CHUNK_SIZES}, CHUNK_OVERLAPS: {CHUNK_OVERLAPS}...')

    for chunk_size, chunk_overlap in (pbar := tqdm(CHUNK_VARIATIONS)):
        folder_name = f'{INGEST_BASE_PATH}/{INGEST_NAME}x{chunk_size}x{chunk_overlap}'
        if os.path.isdir(folder_name):
            pbar.update(1)
            continue
        os.mkdir(folder_name)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        docs = text_splitter.split_documents(json_loader.load())

        # Load into chroma
        pbar.set_description(f"{chunk_size}x{chunk_overlap}->{len(docs)}")
        embedding_function = OllamaEmbeddings(model=MODEL_NAME, base_url=BASE_URL)
        db = Chroma.from_documents(docs, embedding_function, persist_directory=folder_name)
        pbar.update(1)
