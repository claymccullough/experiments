from langchain_core.messages import HumanMessage


def gen_eval_response_prompt(question: str, answer: str) -> str:
    return f"""
    You are an expert at deciphering whether a given answer is a full and complete answer to any given question.  Any answer that contains uncertainty, asks to do more research or says answering is impossible based on the given context is not a full and complete answer.

    I will give you a question and answer and I want you to tell me whether or not the answer is a full and complete response to the question.  respond 'YES' or 'NO', don't tell me your reasoning why.

    Question: "{question}"

    Answer: "{answer}"

    Is this a full and complete answer? 
    
    Given the following question:
    {question}

    Does this answer fully answer the question with detail?
    {answer}

    Answer 'YES' or 'NO' on whether or not the answer properly answers the question.  
    Note: answers that reflect uncertainty, lack of knowledge, or not having the information needed to answer the 
    question, or there being no explicit information about the question, or there being no explicit mention of 
    information asked for, or the provided context does not contain information about the question, are not proper answers.

    Answer:"""

def eval_response(llm, question: str, answer: str):
    # TODO: eval LLM response to question, make sure it makes sense.
    prompt = gen_eval_response_prompt(question=question, answer=answer)
    response = llm.invoke([
        HumanMessage(content=prompt)
    ])
    return response and response.content and response.content.upper() == 'YES'