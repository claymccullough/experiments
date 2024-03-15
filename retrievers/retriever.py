import os
from typing import List, Any

import requests
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.callbacks import CallbackManagerForRetrieverRun, AsyncCallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

"""
GLOBALS
"""
DEST_PATH = os.environ.get('DEST_PATH')
EMBED_MODEL_NAME = os.environ.get('EMBED_MODEL_NAME')
EMBED_BASE_URL = os.environ.get('EMBED_BASE_URL')
COMPRESS_BASE_URL = os.environ.get('COMPRESS_BASE_URL')


class MyRetriever(VectorStoreRetriever):
    def _get_relevant_documents(
            self, query: str, *args, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        documents = super()._get_relevant_documents(query, run_manager=run_manager)
        return documents

    async def _aget_relevant_documents(
            self,
            query: str,
            *,
            run_manager: AsyncCallbackManagerForRetrieverRun,
            **kwargs: Any,
    ) -> List[Document]:
        raise NotImplementedError()


def get_embeddings():
    return OllamaEmbeddings(model=EMBED_MODEL_NAME, base_url=EMBED_BASE_URL)


def get_vector(file_path=DEST_PATH):
    # Define embedding function
    embedding_function = get_embeddings()
    return Chroma(persist_directory=file_path, embedding_function=embedding_function)


def get_vector_retriever(file_path=DEST_PATH, k=4):
    # # Define a retriever interface
    # return MyRetriever(
    #     vectorstore=get_vector(file_path=file_path),
    #     search_type="similarity",
    #     search_kwargs={'k': 4}
    # )
    return get_vector(file_path=file_path).as_retriever(search_kwargs={'k': k})


def compress_prompt(req_json={}):
    result = requests.post(f'{COMPRESS_BASE_URL}/prompt_compressor', json=req_json)
    return result.json()
