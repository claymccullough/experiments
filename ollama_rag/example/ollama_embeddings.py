from langchain.embeddings import OllamaEmbeddings

"""
GLOBALS
"""
MODEL_NAME = 'mistral'

if __name__ == '__main__':
    print('Testing OLLAMA embeddings')
    text = 'This is a test document'
    embeddings = OllamaEmbeddings(model=MODEL_NAME)
    query_result = embeddings.embed_query(text)
    print(query_result[:5])

    doc_result = embeddings.embed_documents([text])
    print(doc_result[0][:5])

