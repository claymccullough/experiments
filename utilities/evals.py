from langchain_core.messages import HumanMessage


def eval_response(llm, question: str, answer: str):
    # TODO: eval LLM response to question, make sure it makes sense.
    prompt = f"""
        Given the following question:
        {question}

        Does this answer properly answer the question:
        {answer}

        Answer 'YES' or 'NO' on whether or not the answer properly answers the question.  
        Note: answers that reflect uncertainty, lack of knowledge, or not having the information needed to answer the 
        question, or there being no explicit information about the question, or there being no explicit mention of 
        information asked for, or the provided context does not contain information about the question, are not proper answers.

        Answer:"""
    response = llm.invoke([
        HumanMessage(content=prompt)
    ])
    return response and response.content and response.content.upper() == 'YES'