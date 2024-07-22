import glob
import json
import os
from typing import List, Any

import structlog
from langchain import hub
from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_experimental.text_splitter import SemanticChunker
from tqdm import tqdm
from deepeval import metrics, evaluate
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from schema.schema import Persona, Trait, Configuration

logger = structlog.get_logger()
EMBED_BASE_URL = "http://localhost:11435"
EMBED_MODEL_NAME = "nomic-embed-text"
INFER_BASE_URL = "http://localhost:11434"
INFER_MODEL_NAME = "llama3:instruct"
DEFAULT_THRESHOLD = 0.7


persona = Persona(
    name="football coach",
    traits=[
        Trait(
            question="What is your strongest skill?",
            answer="I can motivate my players to do far beyond what they thought possible"
        ),
        Trait(
            question="What kind of football plays do you like to run?",
            answer="I prefer to run passing plays. While they are more risky, they have greater potential for reward and don't wear my team out as much."
        )
    ]
)

configuration = Configuration()


def extract_questions(content: str) -> List[dict]:
    try:
        return json.loads(content)
    except Exception as e:
        logger.warning(f"Could not extract questions from content: {e}")
        return []

def gen_prompt(content: str) -> str:
    json_format = """
    [
      {
        "id": 1,
        "question": "Who won the Super Bowl?",
        "answer": "YOUR_ANSWER",
        "ground_truth": "text from context that backs it up"
      }
    ]
    """
    logger.info(f"Question: {content}")

    return f"""
    You are an expert at extracting questions a document can answer.

    Given this context below:
    {content}
    
    I want you to respond in valid json list format.
    example:
    {json_format}

    Do not respond with anything other than the response json. Tell me the questions this context can answer with ground truth to back it up:"""


class OllamaModel(DeepEvalBaseLLM):
    def __init__(
            self,
            model
    ):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content

    def get_model_name(self):
        return "Custom Ollama Model"

def evaluate_response(question, answer, ground_truth):
    llm = ChatOllama(
        model=INFER_MODEL_NAME,
        base_url=INFER_BASE_URL,
        temperature=0.0,
    )
    model = OllamaModel(llm)
    default_args = {
        'model': model,
        'threshold': DEFAULT_THRESHOLD,
        'include_reason': True
    }
    answer_relevancy_metric = metrics.AnswerRelevancyMetric(**default_args)
    answer_faithfulness_metric = metrics.FaithfulnessMetric(**default_args)
    hallucination_metric = metrics.HallucinationMetric(**default_args)

    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        context=[ground_truth],
        expected_output=ground_truth,
        retrieval_context=[ground_truth]
    )
    test_results = evaluate([test_case], [
        answer_relevancy_metric,
        answer_faithfulness_metric,
        hallucination_metric,
    ])
    return test_results


def gen_memory_retriever(documents) -> VectorStoreRetriever:
    embedding_function = OllamaEmbeddings(model=EMBED_MODEL_NAME, base_url=EMBED_BASE_URL)
    return Chroma.from_documents(documents, embedding_function).as_retriever()

def gen_questions(documents, persona):
    file_path = "./gen/questions.json"
    if os.path.isfile(file_path):
        with open(file_path) as f:
            return json.load(f)

    llm = ChatOllama(
        model=INFER_MODEL_NAME,
        base_url=INFER_BASE_URL,
        temperature=0.0,
    )
    final_questions = []
    for document in tqdm(documents[:3]):
        # TODO: personalize questions
        response = llm.invoke(gen_prompt(content=str(document)))
        questions = extract_questions(response.content)
        if len(questions) < 1:
            logger.info("No new questions, continuing...")
            continue
        logger.info(f"Found {len(questions)} new questions!")
        for result in questions:
            logger.info(result)
        # use RAG to ignore similar questions
        final_questions.extend(questions)

    # TODO: Eval step of questions

    with open(file_path, "w") as f:
        json.dump(final_questions, f)

    return final_questions


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def load_questions(file_root: str) -> List[dict]:
    questions = set()
    for file_path in glob.glob(os.path.join(file_root, "*.json")):
        try:
            with open(file_path) as f:
                result = json.load(f)
            if not result:
                logger.warning(f"{file_path}: null result")

            for question_data in result:
                questions.add(question_data["question"])
        except Exception as e:
            logger.warning(f"Could not load questions from {file_path}: {e}")
    return list(questions)

if __name__ == '__main__':
    logger.info('Starting Gen')

    # TODO: Step 0, get configuration
    logger.info(persona.json())

    # TODO: Step 1, embed input data
    loader = PyPDFLoader("./gen/Super_Bowl_LVIII.pdf")
    docs = loader.load()

    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    # splits = text_splitter.split_documents(docs)
    # logger.info(f"splits={len(splits)}")

    text_splitter = SemanticChunker(OllamaEmbeddings(model=EMBED_MODEL_NAME, base_url=EMBED_BASE_URL))
    splits = text_splitter.split_documents(docs)
    logger.info(f"splits={len(splits)}")

    # TODO: Step 2, Gen questions we can ask about the data
    questions = gen_questions(documents=splits, persona=persona)
    logger.info(questions)

    # # Load the vector DB
    # embedding_function = OllamaEmbeddings(model=EMBED_MODEL_NAME, base_url=EMBED_BASE_URL)
    # # db = Chroma.from_documents(splits, embedding_function, persist_directory="./chroma_db")
    # # retriever = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function).as_retriever()
    #
    # # TODO: Step 2, Gen questions we can ask about the data
    # llm = ChatOllama(
    #     model=INFER_MODEL_NAME,
    #     base_url=INFER_BASE_URL,
    #     temperature=0.0,
    # )
    #
    # file_root = "./gen/output"
    # i = 0
    # for split in tqdm(splits):
    #     response = llm.invoke(gen_prompt(content=str(split)))
    #     questions = extract_questions(response.content)
    #     with open(f"{file_root}/{i}.json", "w") as f:
    #         json.dump(questions, f, indent=4)
    #     i += 1
    #
    # # Get the questions loaded up.
    # # questions  = load_questions(file_root=file_root)
    # # logger.info(questions)
    # # with open(f"{file_root}/questions.json", "w") as f:
    # #     json.dump(questions, f, indent=4)
    #
    # # TODO: Step 3, Gen answers to questions via RAG
    #
    # prompt = hub.pull("rlm/rag-prompt")
    # rag_chain = (
    #     {"context": retriever | format_docs, "question": RunnablePassthrough()}
    #     | prompt
    #     | llm
    #     | StrOutputParser()
    # )
    # # questions = load_questions(file_root)[:10]
    # # qa = []
    # # for question in tqdm(questions):
    # #     # logger.info(f"question={question}")
    # #     response = rag_chain.invoke(question)
    # #     # logger.info(f"answer={response}")
    # #     context = retriever.invoke(question)
    # #     qa.append({"question": question, "answer": response, "context": [doc.page_content for doc in context]})
    # #
    # # with open("./gen/output/qa_context.json", "w") as f:
    # #     json.dump(qa, f, indent=4)
    #
    # # TODO: Step 4, Eval answers to questions, RAG performance
    #
    # # TODO: Step 5, Output
    #