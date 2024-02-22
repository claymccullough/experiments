from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings

load_dotenv('.env')

import os

from langchain_community.vectorstores.chroma import Chroma

MODEL_NAME = os.environ.get("MODEL_NAME")

if __name__ == '__main__':
    print('RAG')
    # load from disk
    query = 'What does Limeade use for a message bus'
    embedding_function = OllamaEmbeddings(model=MODEL_NAME)
    db3 = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
    docs = db3.similarity_search(query)
    print(len(docs))
    print(docs)
