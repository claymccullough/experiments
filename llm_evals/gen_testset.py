from langchain.text_splitter import TokenTextSplitter
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import BaseRagasLLM, LangchainLLMWrapper
from ragas.testset import TestsetGenerator
from ragas.testset.docstore import InMemoryDocumentStore
from ragas.testset.extractor import KeyphraseExtractor

from llm_evals.ingest import get_docs, get_embeddings
from llm_evals.rag_agent import get_ollama_model

if __name__ == '__main__':
    llm = LangchainLLMWrapper(get_ollama_model())
    docs = get_docs()
    embeddings = LangchainEmbeddingsWrapper(get_embeddings())
    keyphrase_extractor = KeyphraseExtractor(llm=llm)
    splitter = TokenTextSplitter(chunk_size=4000, chunk_overlap=0)
    docstore = InMemoryDocumentStore(
        splitter=splitter,
        embeddings=embeddings,
        extractor=keyphrase_extractor,
    )
    generator = TestsetGenerator(
        generator_llm=llm,
        critic_llm=llm,
        embeddings=embeddings,
        docstore=docstore
    ).generate_with_langchain_docs(
        documents=docs,
        test_size=10,
        distributions={'simple': 0.5, 'reasoning': 0.25, 'multi_context': 0.25}
    )
