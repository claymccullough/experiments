import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from langchain.evaluation import load_evaluator, EvaluatorType, CriteriaEvalChain, Criteria, LabeledCriteriaEvalChain
from langchain_community.llms.ollama import Ollama

from llm_evals.rag_agent import get_rag_agent, get_ollama_model, get_vector_retriever


class OllamaModel(DeepEvalBaseLLM):
    def __init__(
            self,
            model
    ):
        self.model = model

    def load_model(self):
        return self.model

    def _call(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.invoke(prompt)

    def get_model_name(self):
        return "Custom Ollama Model"



if __name__ == '__main__':
    query = "What is the population of Boston"

    agent, stats_handler = get_rag_agent()
    retriever = get_vector_retriever()
    llm = get_ollama_model()

    # build the context
    docs = retriever.get_relevant_documents(query)
    context = [doc.page_content for doc in docs]
    result = agent.invoke(query)

    test = 'hi'

    evaluator = LabeledCriteriaEvalChain.from_llm(llm=llm, criteria=Criteria.CONCISENESS)
    eval_result = evaluator.evaluate_strings(
        input=query,
        prediction=result['output'],
        reference=' '.join(context)
    )
    print(eval_result)

# def test_case():
#     query = "What is the population of New York City?"
#
#     agent, stats_handler = get_rag_agent()
#     retriever = get_vector_retriever()
#     llm = get_ollama_model()
#
#     # build the context
#     docs = retriever.get_relevant_documents(query)
#     context = [doc.page_content for doc in docs]
#     result = agent.invoke(query)
#
#     test = 'hi'
#
#     evaluator = CriteriaEvalChain.from_llm(llm=llm, criteria='correctness')
#     eval_result = evaluator.evaluate_strings(
#         input=query,
#         prediction=result['output'],
#         reference=' '.join(context)
#     )
#     print(eval_result)

    # # query = 'What is the population of Houston, TX?'
    # # actual_output = ''
    # #
    # answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5, model=llm)
    # test_case = LLMTestCase(
    #     input=query,
    #     # Replace this with the actual output from your LLM application
    #     actual_output=result['output'],
    #     retrieval_context=context
    # )
    # assert_test(test_case, [answer_relevancy_metric])
