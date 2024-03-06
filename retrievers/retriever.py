import os

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores.chroma import Chroma

"""
GLOBALS
"""
DEST_PATH = os.environ.get('DEST_PATH')
EMBED_MODEL_NAME = os.environ.get('EMBED_MODEL_NAME')
EMBED_BASE_URL = os.environ.get('EMBED_BASE_URL')


def get_embeddings():
    return OllamaEmbeddings(model=EMBED_MODEL_NAME, base_url=EMBED_BASE_URL)


def get_vector_retriever(file_path=DEST_PATH):
    # Define embedding function
    embedding_function = get_embeddings()
    vector = Chroma(persist_directory=file_path, embedding_function=embedding_function)

    # Define a retriever interface
    return vector.as_retriever()
