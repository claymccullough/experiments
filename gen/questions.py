import json
from typing import List

from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from tqdm import tqdm

load_dotenv(".env")

import structlog
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

logger = structlog.get_logger()
EMBED_BASE_URL = "http://localhost:11435"
EMBED_MODEL_NAME = "nomic-embed-text"
INFER_BASE_URL = "http://localhost:11434"
INFER_MODEL_NAME = "llama3:instruct"
DEFAULT_THRESHOLD = 0.7

json_format = """[
    {
    "id": 1,
    "question": "Who won the Super Bowl?",
    "answer": "YOUR_ANSWER",
    "ground_truth": "text from context that backs it up"
    }
]"""

def extract_questions(content: str) -> List[dict]:
    try:
        return json.loads(content)
    except Exception as e:
        logger.warning(f"Could not extract questions from content: {e}")
        return []

if __name__ == '__main__':
    logger.info("Generate questions from a dataset, just provide a filename and gen does the rest.")
    file_name = "./gen/Super_Bowl_LVIII.pdf"

    # TODO: support any input format, just PDF for now
    loader = PyPDFLoader(file_name)
    docs = loader.load()

    # TODO: support any splitter, semantic, etc.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", ";"],  # Order matters
        keep_separator=True,
    )
    # text_splitter = SemanticChunker(OllamaEmbeddings(model=EMBED_MODEL_NAME, base_url=EMBED_BASE_URL))
    splits = text_splitter.split_documents(docs)
    logger.info(f"Number splits: {len(splits)}")

    # TODO: support any model
    llm = ChatOpenAI(
        verbose=True,
        temperature=0.0,
        model_name="gpt-3.5-turbo",
    )
    final_questions = []

    # Iteration 1, look at the text
    # TODO: Iteration 2: Combine questions and answers, ask higher-level questions
    for split in tqdm(splits[:10]):
        # TODO: extract facts first

        # TODO: then generate question(s) based on those facts.


        prompt = f"""
        You are an expert at extracting questions text can answer. Given the context in triple backticks, I want you to respond in valid json list format.
        example:
        {json_format}

        Do not respond with anything other than the response json. Tell me the questions this context can answer with ground truth to back it up.

        Context: ```{split.page_content}```"""
        logger.info(prompt)
        # response = llm.invoke(prompt)
        # logger.info(f"Content: {split}")
        # logger.info(f"Response: {response}")
        # questions = extract_questions(response.content)
        # logger.info(f"Extracted questions: {len(questions)}")
        # final_questions.extend(questions)
        #
        # out_file_name = "./gen/final_questions.json"
        # logger.info(f"Dumping to file: {out_file_name}")
        # with open(out_file_name, "w") as f:
        #     json.dump(final_questions, f, indent=4)

    logger.info(f"DONE")
