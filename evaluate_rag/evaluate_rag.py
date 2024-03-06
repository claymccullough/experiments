import json
import os

import requests
from tqdm import tqdm

from evaluations.evaluate import evaluate_response
from ingest.ingest import CHUNK_VARIATION_FILES
from models.models import get_ollama_model
from retrievers.retriever import get_vector_retriever

"""
GLOBALS
"""
INFER_MODEL_NAME = os.environ.get("INFER_MODEL_NAME")
EVALUATION_FILE_NAME = "./evaluate_rag/evaluations/evaluation.json"
COMPRESS_BASE_URL = os.environ.get('COMPRESS_BASE_URL')


def generate_cases():
    llm, stats_handler = get_ollama_model()
    return [
        {
            'instruction': instruction,
            'question': question,
            'ground_truth': ground_truth,
            'embedding_file_path': embedding_file_path,
            'retriever': get_vector_retriever(file_path=embedding_file_path),
            'compression_ratio': compression_ratio,
            'compression_target': compression_target,
            'llm': llm,
            'stats_handler': stats_handler
        }
        for instruction in [
            """Use only information provided below without other sources.  If you don't know the answer, say "I don't know"."""]
        for question in ['What is the population of Houston, TX?']
        for ground_truth in ['The population of Houston, TX is 2,302,878.']
        for embedding_file_path in CHUNK_VARIATION_FILES
        for compression_ratio in [None]
        for compression_target in [None]
    ]


def evaluate_case(instruction, question, ground_truth, embedding_file_path, retriever, compression_ratio, compression_target, llm,
                  stats_handler) -> dict:
    docs = retriever.get_relevant_documents(question)
    raw_context = [doc.page_content for doc in docs]
    context = "\n\n".join(raw_context)
    if compression_ratio and compression_target:
        result = requests.post(f'{COMPRESS_BASE_URL}/prompt_compressor', json={
            "context": raw_context,
            "instruction": instruction,
            "question": question,
            "ratio": compression_ratio,
            "target_token": compression_target
        }).json()
        prompt = result['prompt']
    else:
        prompt = f"""
        {instruction}
        
        {context}
        
        {question}
        """

    output = llm.invoke(prompt)
    stats = stats_handler.get_stats()
    stats_handler.reset()

    """
    Metric evaluation step
    """
    test_results = evaluate_response(
        question=question,
        response=output,
        ground_truth=ground_truth,
        context=context,
    )

    final = {
        'case': {
            'model_name': INFER_MODEL_NAME,
            'input': prompt,
            'output': output,
            'instruction': instruction,
            'question': question,
            'context': context,
            'compression_ratio': compression_ratio,
            'compression_target': compression_target,
            'embedding_file_path': embedding_file_path,
            'model': INFER_MODEL_NAME,
        },
        'token_metrics': stats,
        'evaluation_results': {
            metric.__name__: {
                'evaluation_model': metric.evaluation_model,
                'score': metric.score,
                'reason': metric.reason,
                'threshold': metric.threshold
            } for metric in test_results[0].metrics
        }
    }
    return final


if __name__ == '__main__':
    """
    EVALUATING RAG:
    Consider:
    - Different questions
    - Different embeddings
    - Different prompts
    - Different Prompt compression (different thresholds) or not at all
    - Different models (mistral vs GPT)
    
    = What accuracy?
    - Relevancy
    - Contextual Precision
    - Contextual Relevancy
    - Contextual Recall
    - Hallucination
    - Input Start Tokens
    - Input Total Tokens
    - Output Final Tokens
    - Output Total Tokens
    - RAG Latency
    """
    print('EVALUATING RAG')
    cases = generate_cases()
    results = []
    for case in tqdm(cases):
        result = evaluate_case(**case)
        results.append(result)

    with open(EVALUATION_FILE_NAME, 'w') as f:
        json.dump(results, f, indent=4, sort_keys=True)

    print('EVAL COMPLETE')
