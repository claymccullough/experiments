from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI

from utilities.evals import eval_response

load_dotenv('.env')
EVAL_OPENAI_MN = os.environ.get('EVAL_OPENAI_MN')

test_cases = [
    {
        "question": 'How long is the model.limeade_lakehouse.sil_ai_identity_events model set to retain when it VACUUMs?',
        "answer": 'The provided context does not contain information about the retention period for the model.limeade_lakehouse.sil_ai_identity_events model when it VACUUMs. I recommend checking the specific VACUUM configuration for this model in the data warehouse or consulting the documentation for the data retention policies related to VACUUM operations.',
        "correct_response": False
    },
    {
        "question": 'On the model.limeade_lakehouse.gold_dim_dates model, what type is the is_last_day_of_year column?',
        "answer": 'I don\'t have the information on the model.limeade_lakehouse.gold_dim_dates model.',
        "correct_response": False
    }

]

if __name__ == '__main__':
    utility_llm = ChatOpenAI(
        verbose=True,
        temperature=0.0,
        model_name=EVAL_OPENAI_MN,
        # streaming=True
    )
    for test_case in test_cases:
        question, answer, correct_response = test_case['question'], test_case['answer'], test_case["correct_response"]
        llm_response = eval_response(
            llm=utility_llm,
            question=question,
            answer=answer
        )
        print(f'LLM: {llm_response}, CORRECT: {correct_response}')
        assert llm_response == correct_response
