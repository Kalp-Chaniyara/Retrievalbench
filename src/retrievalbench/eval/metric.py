import asyncio

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)
from deepeval.metrics.base_metric import BaseMetric
from deepeval.test_case import LLMTestCase

from retrievalbench.model import EvalScores, MetricScore, RetrievedChunk


async def _measure(metric: BaseMetric, test_case: LLMTestCase) -> MetricScore:
    await metric.a_measure(test_case)
    return MetricScore(score=metric.score, reason=metric.reason)


async def evaluate_query(
    model: str,
    query: str,
    response: str,
    expected_answer: str,
    retrieved_chunks: list[RetrievedChunk],
) -> EvalScores:
    """Score one query against all four DeepEval RAG metrics.

    - faithfulness: does the answer stay grounded in the retrieved context?
    - answer_relevancy: does the answer actually address the query?
    - context_precision: are the relevant chunks ranked above the noise?
    - context_recall: does the retrieved context cover the expected answer?
    """
    test_case = LLMTestCase(
        input=query,
        actual_output=response,
        expected_output=expected_answer,
        retrieval_context=[chunk.text for chunk in retrieved_chunks],
    )

    # Independent LLM-judge calls -> run them concurrently, not in a loop.
    faith, relevancy, precision, recall = await asyncio.gather(
        _measure(FaithfulnessMetric(model=model, include_reason=True), test_case),
        _measure(AnswerRelevancyMetric(model=model, include_reason=True), test_case),
        _measure(
            ContextualPrecisionMetric(model=model, include_reason=True), test_case
        ),
        _measure(ContextualRecallMetric(model=model, include_reason=True), test_case),
    )

    return EvalScores(
        faithfulness=faith,
        answer_relevancy=relevancy,
        context_precision=precision,
        context_recall=recall,
    )
