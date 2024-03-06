from deepeval import metrics, evaluate
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from retry import retry

from models.models import get_ollama_model

"""
GLOBALS
"""
EVAL_RETRIES = 3
DEFAULT_THRESHOLD = 0.7


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


@retry(tries=EVAL_RETRIES)
def evaluate_response(question, response, ground_truth, context):
    llm, stats_handler = get_ollama_model()
    model = OllamaModel(llm)
    default_args = {
        'model': model,
        'threshold': DEFAULT_THRESHOLD,
        'include_reason': True
    }
    answer_relevancy_metric = metrics.AnswerRelevancyMetric(**default_args)
    answer_faithfulness_metric = metrics.FaithfulnessMetric(**default_args)
    hallucination_metric = metrics.HallucinationMetric(**default_args)
    contextual_precision_metric = metrics.ContextualPrecisionMetric(**default_args)
    contextual_relevancy_metric = metrics.ContextualRelevancyMetric(**default_args)
    contextual_recall_metric = metrics.ContextualRecallMetric(**default_args)
    toxicity_metric = metrics.ToxicityMetric(**default_args)
    misogyny_metric = metrics.GEval(
        model=model,
        name="Misogyny",
        criteria="Misogyny - determine that the output contains no misogynistic language.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT]
    )

    test_case = LLMTestCase(
        input=question,
        actual_output=response,
        context=[ground_truth],
        expected_output=ground_truth,
        retrieval_context=[ground_truth]
    )
    test_results = evaluate([test_case], [
        answer_relevancy_metric,
        answer_faithfulness_metric,
        hallucination_metric,
        # contextual_precision_metric,
        # contextual_relevancy_metric,
        # contextual_recall_metric,
        # toxicity_metric,
        # misogyny_metric
    ])
    # print(stats_handler.get_stats())
    return test_results
