import httpx
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)
from deepeval.metrics.base_metric import BaseMetric
from deepeval.models import GPTModel
from deepeval.test_case import LLMTestCase

from retrievalbench.model import EvalScores, MetricScore, RetrievedChunk

# Why we own the HTTP client instead of letting deepeval make its own:
# GPTModel._build_client() constructs a BRAND-NEW AsyncOpenAI — and therefore a
# fresh httpx connection pool — on EVERY LLM call, and never closes it. Scoring
# one query opened 12 separate TLS connections with zero reuse. Establishing a
# new connection intermittently stalls: the socket connects, the request is sent
# (and billed), and the response never arrives, so the call hangs until the
# per-attempt timeout and then retries onto another cold connection — which is
# how a 4-second call turned into a ~6-minute RetryError. Handing every metric
# ONE pooled client keeps connections warm and reused; the call that had been
# hanging indefinitely then succeeded 8/8 at ~3s.
DEFAULT_MAX_CONNECTIONS = 8
# A ceiling, not a target: real calls land in 1-9s. This exists so a stalled
# connection is abandoned and retried in seconds instead of hanging. Deepeval's
# DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE sits *above* this.
DEFAULT_HTTP_TIMEOUT_S = 60.0


async def _measure(metric: BaseMetric, test_case: LLMTestCase) -> MetricScore:
    # _show_indicator=False: deepeval otherwise opens its own rich.Progress per
    # metric, which stacks 4 extra Live displays on top of runner.py's progress
    # bar. Cosmetic (it was NOT the cause of the hangs) but they collide.
    await metric.a_measure(test_case, _show_indicator=False)
    return MetricScore(score=metric.score or 0.0, reason=metric.reason or "")


class Scorer:
    """Scores queries with the four DeepEval RAG metrics over one shared,
    pooled HTTP client and one judge model.

    Config (judge model, pool limits, timeout) binds at construction; the
    query/answer/context vary per call. Owns the httpx client, so callers must
    `await aclose()` when done — `runner.py` does this in a finally.

    The judge model is exposed as `.model` so the diagnostics engine's
    correctness trigger scores through the SAME pooled connections rather than
    reintroducing per-call clients.
    """

    def __init__(
        self,
        model: str,
        *,
        max_connections: int = DEFAULT_MAX_CONNECTIONS,
        timeout_s: float = DEFAULT_HTTP_TIMEOUT_S,
    ) -> None:
        self._http = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_connections,
            ),
            timeout=httpx.Timeout(timeout_s, connect=10.0),
        )
        self.model = GPTModel(model=model, async_http_client=self._http)

    async def warmup(self) -> None:
        """Establish one pooled connection before the first concurrent burst,
        so the 4 metrics reuse a warm connection instead of racing to open 4
        cold ones (cold-connection setup is exactly what stalls)."""
        await self.model.a_generate("Reply with the single word: ok")

    async def evaluate_query(
        self,
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

        # Sequential, NOT asyncio.gather — a deliberate reversal.
        #
        # These calls are independent, so concurrency looks right and was the
        # original design. But the judge (gpt-4o) is capped at 30k TOKENS per
        # minute, and each metric spends ~5k. Firing four at once bursts past
        # the cap, tenacity exhausts its retries, and the whole run dies with a
        # RetryError. Wall time is unchanged either way — TPM is the ceiling,
        # not latency — so serializing costs nothing and removes the burst.
        faith = await _measure(
            FaithfulnessMetric(model=self.model, include_reason=True), test_case
        )
        relevancy = await _measure(
            AnswerRelevancyMetric(model=self.model, include_reason=True), test_case
        )
        precision = await _measure(
            ContextualPrecisionMetric(model=self.model, include_reason=True), test_case
        )
        recall = await _measure(
            ContextualRecallMetric(model=self.model, include_reason=True), test_case
        )

        return EvalScores(
            faithfulness=faith,
            answer_relevancy=relevancy,
            context_precision=precision,
            context_recall=recall,
        )

    async def aclose(self) -> None:
        await self._http.aclose()
