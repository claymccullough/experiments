import os

from dotenv import load_dotenv

load_dotenv('.env')
import json

from dbt_rag import gen_prompt
from models.models import get_ollama_model
from retrievers.retriever import compress_prompt

"""
Seems well-tuned for metadata queries.
dbt-test is Metadata
Now, get a retriever for the columns.
"""
TEST_NAME = os.environ.get('TEST_NAME')
COMPRESSED = True
FILE_NAME = f'./dbt_rag/dbt_{TEST_NAME}_compressed_output.json' if COMPRESSED else f'./dbt_rag/dbt_{TEST_NAME}_output.json'

if __name__ == '__main__':
    print('DBT TEST RAG')
    llm, stats_handler = get_ollama_model()
    with open(f'./dbt_rag/dbt_{TEST_NAME}_test.json') as f:
        rag_test = json.load(f)

    results = []
    for item in rag_test:
        question = item['question']
        answer = item['answer']
        result = gen_prompt(question)
        prompt = result['prompt']
        context = result['context']
        compressed = {}
        if COMPRESSED:
            compressed = compress_prompt({
                "context": context,
                "instruction": prompt,
                "question": question,
                "ratio": 0.3,
            })
        prompt = compressed['compressed_prompt'] if COMPRESSED else prompt
        response = llm.invoke(prompt)
        results.append({
            **item,
            'prompt': prompt,
            'response': response,
            'stats': {
                **stats_handler.get_stats(),
                **compressed
            }
        })
        stats_handler.reset()

    with open(FILE_NAME, 'w') as f:
        json.dump(results, f, indent=4)

    print('DONE')
