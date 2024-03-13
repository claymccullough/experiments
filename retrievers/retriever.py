import os

import requests
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores.chroma import Chroma

"""
GLOBALS
"""
DEST_PATH = os.environ.get('DEST_PATH')
EMBED_MODEL_NAME = os.environ.get('EMBED_MODEL_NAME')
EMBED_BASE_URL = os.environ.get('EMBED_BASE_URL')
COMPRESS_BASE_URL = os.environ.get('COMPRESS_BASE_URL')


def get_embeddings():
    return OllamaEmbeddings(model=EMBED_MODEL_NAME, base_url=EMBED_BASE_URL)


def get_vector(file_path=DEST_PATH):
    # Define embedding function
    embedding_function = get_embeddings()
    return Chroma(persist_directory=file_path, embedding_function=embedding_function)


def get_vector_retriever(file_path=DEST_PATH):
    # Define a retriever interface
    return get_vector(file_path=file_path).as_retriever(search_kwargs={'k': 4})


def compress_prompt(req_json={}):
    result = requests.post(f'{COMPRESS_BASE_URL}/prompt_compressor', json=req_json)
    return result.json()
