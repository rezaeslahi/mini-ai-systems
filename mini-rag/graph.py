from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Any, Literal, Optional, TypedDict, cast
import operator

from langgraph.graph import START, END, StateGraph


# ============================================================
# Domain models
# ============================================================

class IntentName(str, Enum):
    GENERAL = "general"
    PRIMARY_PARAMETER = "primary_parameter"
    SECONDARY_PARAMETER = "secondary_parameter"
    PLAN_ASSESSOR = "plan_assessor"


class DataSource(str, Enum):
    STRUCTURED_DB = "structured_db"
    DOCUMENTS = "documents"


class SqlReviewStatus(str, Enum):
    NEEDS_REVISION = "needs_revision"
    APPROVED = "approved"


@dataclass(frozen=True)
class IntentPrediction:
    source: str
    intents: list[IntentName]
    confidence: float
    rationale: str


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    source_document: str


@dataclass(frozen=True)
class SqlDraft:
    sql: str
    explanation: str
    target_database: str


@dataclass(frozen=True)
class SqlCritique:
    status: SqlReviewStatus
    issues: list[str]
    suggested_sql: Optional[str]


@dataclass(frozen=True)
class QueryResultRow:
    values: dict[str, Any]


@dataclass(frozen=True)
class SecondaryComputationResult:
    metric_name: str
    value: Any
    unit: Optional[str]
    explanation: str


@dataclass(frozen=True)
class PlanAssessmentResult:
    feasible: bool
    summary: str
    evidence: list[str]


@dataclass(frozen=True)
class FinalAnswer:
    text: str
    citations: list[str]
    chart_spec: Optional[dict[str, Any]]


# ============================================================
# Reducers
# ============================================================

def merge_intent_predictions(
    left: list[IntentPrediction],
    right: list[IntentPrediction],
) -> list[IntentPrediction]:
    return left + right


def merge_retrieved_chunks(
    left: list[RetrievedChunk],
    right: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    return left + right


def merge_rows(
    left: list[QueryResultRow],
    right: list[QueryResultRow],
) -> list[QueryResultRow]:
    return left + right


def merge_strings(
    left: list[str],
    right: list[str],
) -> list[str]:
    return left + right


# ============================================================
# Graph state
# ============================================================

class GraphState(TypedDict, total=False):
    # Core request
    user_question: str
    conversation_id: str

    # Intent stage
    intent_predictions: Annotated[list[IntentPrediction], merge_intent_predictions]
    final_intents: list[IntentName]

    # RAG stage
    retrieved_chunks: Annotated[list[RetrievedChunk], merge_retrieved_chunks]
    filtered_chunks: list[RetrievedChunk]

    # SQL generation / review loop
    sql_needed: bool
    sql_draft: Optional[SqlDraft]
    sql_critique: Optional[SqlCritique]
    sql_review_iteration: int
    max_sql_review_iterations: int
    sql_final: Optional[str]

    # SQL execution
    primary_rows: Annotated[list[QueryResultRow], merge_rows]
    primary_result_summary: Optional[str]

    # Secondary calculation
    secondary_needed: bool
    secondary_metric_name: Optional[str]
    secondary_result: Optional[SecondaryComputationResult]

    # Plan assessment
    plan_assessor_needed: bool
    plan_assessment_result: Optional[PlanAssessmentResult]

    # Final response
    references: Annotated[list[str], merge_strings]
    final_answer: Optional[FinalAnswer]


# ============================================================
# LLM abstraction
# ============================================================

class LLMClient:
    async def ainvoke(self, prompt: str) -> str:
        # Placeholder only.
        # Replace with your actual local or remote LLM call.
        raise NotImplementedError("Implement your LLM backend here.")


@dataclass(frozen=True)
class LLMRegistry:
    intent_llm: LLMClient
    intent_aggregator_llm: LLMClient
    sql_generator_llm: LLMClient
    sql_critic_llm: LLMClient
    final_answer_llm: LLMClient


# ============================================================
# Short prompt builders
# ============================================================

def build_intent_prompt(user_question: str) -> str:
    return (
        "You are an intent classifier for an oil-and-gas data platform. "
        "Classify the question into zero or more of these intents only: "
        "[general, primary_parameter, secondary_parameter, plan_assessor]. "
        f"Question: {user_question}"
    )


def build_intent_aggregator_prompt(
    user_question: str,
    predictions: list[IntentPrediction],
) -> str:
    return (
        "You are an intent aggregation agent. "
        "Given several intent predictions, output the final list of intents. "
        "Always include 'general'. "
        f"Question: {user_question}\n"
        f"Predictions: {predictions}"
    )


def build_sql_generation_prompt(
    user_question: str,
    final_intents: list[IntentName],
) -> str:
    return (
        "You generate SQL for an oil-and-gas structured data platform. "
        "Use only the business need expressed in the question. "
        "Return precise SQL and a short explanation. "
        f"Question: {user_question}\n"
        f"Intents: {[intent.value for intent in final_intents]}"
    )


def build_sql_critic_prompt(
    user_question: str,
    sql_draft: SqlDraft,
) -> str:
    return (
        "You are a SQL critic. Review the SQL for correctness, completeness, "
        "and alignment with the user question. "
        "If correct, approve it. Otherwise suggest a better SQL. "
        f"Question: {user_question}\n"
        f"SQL draft: {sql_draft.sql}"
    )


def build_final_answer_prompt(
    user_question: str,
    filtered_chunks: list[RetrievedChunk],
    primary_rows: list[QueryResultRow],
    secondary_result: Optional[SecondaryComputationResult],
    plan_assessment_result: Optional[PlanAssessmentResult],
) -> str:
    return (
        "You are an oil-and-gas domain assistant. "
        "Answer clearly, professionally, and with engineering-style reasoning. "
        "Use retrieved document context and structured results when available. "
        f"Question: {user_question}\n"
        f"Filtered document chunks: {filtered_chunks}\n"
        f"Primary rows: {primary_rows}\n"
        f"Secondary result: {secondary_result}\n"
        f"Plan assessment: {plan_assessment_result}"
    )


# ============================================================
# Utility parsers
# These are intentionally skeletal.
# ============================================================

def parse_intent_prediction(raw_text: str, source: str) -> IntentPrediction:
    # Placeholder parser.
    # Replace with robust JSON parsing later.
    return IntentPrediction(
        source=source,
        intents=[IntentName.GENERAL],
        confidence=0.50,
        rationale=raw_text,
    )


def parse_aggregated_intents(raw_text: str) -> list[IntentName]:
    # Placeholder parser.
    # Replace with robust JSON parsing later.
    return [IntentName.GENERAL]


def parse_sql_draft(raw_text: str) -> SqlDraft:
    return SqlDraft(
        sql="SELECT 1;",
        explanation=raw_text,
        target_database="production_db",
    )


def parse_sql_critique(raw_text: str) -> SqlCritique:
    return SqlCritique(
        status=SqlReviewStatus.APPROVED,
        issues=[],
        suggested_sql=None,
    )


def parse_final_answer(raw_text: str) -> FinalAnswer:
    return FinalAnswer(
        text=raw_text,
        citations=[],
        chart_spec=None,
    )


# ============================================================
# Tool stubs
# ============================================================

async def semantic_intent_matcher(user_question: str) -> IntentPrediction:
    return IntentPrediction(
        source="semantic_matcher",
        intents=[IntentName.GENERAL],
        confidence=0.70,
        rationale="Nearest examples matched general informational intent.",
    )


def deterministic_intent_detector(user_question: str) -> IntentPrediction:
    lowered: str = user_question.lower()

    detected: list[IntentName] = [IntentName.GENERAL]

    if "water cut" in lowered or "ratio" in lowered:
        detected.append(IntentName.SECONDARY_PARAMETER)

    if "compare plan" in lowered or "feasible" in lowered or "historical" in lowered:
        detected.append(IntentName.PLAN_ASSESSOR)

    if "production" in lowered or "opex" in lowered or "capex" in lowered:
        detected.append(IntentName.PRIMARY_PARAMETER)

    return IntentPrediction(
        source="deterministic",
        intents=detected,
        confidence=0.80,
        rationale="Rule-based keyword detection.",
    )


async def retrieve_rag_chunks(user_question: str, top_k: int) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id="chunk-001",
            text="Example domain/context chunk.",
            score=0.86,
            source_document="operator_report_001.pdf",
        )
    ]


def filter_chunks_by_score(
    chunks: list[RetrievedChunk],
    threshold: float,
) -> list[RetrievedChunk]:
    return [chunk for chunk in chunks if chunk.score >= threshold]


async def execute_sql_query(sql: str) -> list[QueryResultRow]:
    return [
        QueryResultRow(values={"operator": "Operator A", "oil_production": 12000.0}),
        QueryResultRow(values={"operator": "Operator A", "water_production": 3000.0}),
    ]


def summarize_primary_rows(rows: list[QueryResultRow]) -> str:
    return f"Fetched {len(rows)} structured rows."


def detect_secondary_metric(
    user_question: str,
    intents: list[IntentName],
) -> Optional[str]:
    lowered: str = user_question.lower()
    if IntentName.SECONDARY_PARAMETER in intents and "water cut" in lowered:
        return "water_cut"
    return None


def compute_secondary_metric(
    metric_name: str,
    rows: list[QueryResultRow],
) -> SecondaryComputationResult:
    if metric_name == "water_cut":
        oil: float = 0.0
        water: float = 0.0

        for row in rows:
            values: dict[str, Any] = row.values
            oil += float(values.get("oil_production", 0.0))
            water += float(values.get("water_production", 0.0))

        total_liquid: float = oil + water
        value: float = (water / total_liquid) if total_liquid > 0.0 else 0.0

        return SecondaryComputationResult(
            metric_name="water_cut",
            value=value,
            unit="fraction",
            explanation="Computed as water / (oil + water).",
        )

    return SecondaryComputationResult(
        metric_name=metric_name,
        value=None,
        unit=None,
        explanation="Metric not implemented yet.",
    )


def assess_plan_feasibility(
    user_question: str,
    rows: list[QueryResultRow],
    secondary_result: Optional[SecondaryComputationResult],
) -> PlanAssessmentResult:
    return PlanAssessmentResult(
        feasible=True,
        summary="Plan appears broadly consistent with historical behavior.",
        evidence=[
            "Historical trend is within expected range.",
            "No major deviation detected in the current skeleton implementation.",
        ],
    )


# ============================================================
# Nodes
# ============================================================

async def deterministic_intent_node(state: GraphState) -> GraphState:
    question: str = state["user_question"]
    prediction: IntentPrediction = deterministic_intent_detector(question)
    return {"intent_predictions": [prediction]}


async def llm_intent_node(state: GraphState, llms: LLMRegistry) -> GraphState:
    question: str = state["user_question"]
    prompt: str = build_intent_prompt(question)
    raw_text: str = await llms.intent_llm.ainvoke(prompt)
    prediction: IntentPrediction = parse_intent_prediction(raw_text, source="llm")
    return {"intent_predictions": [prediction]}


async def semantic_intent_node(state: GraphState) -> GraphState:
    question: str = state["user_question"]
    prediction: IntentPrediction = await semantic_intent_matcher(question)
    return {"intent_predictions": [prediction]}


async def aggregate_intents_node(state: GraphState, llms: LLMRegistry) -> GraphState:
    question: str = state["user_question"]
    predictions: list[IntentPrediction] = state.get("intent_predictions", [])

    prompt: str = build_intent_aggregator_prompt(question, predictions)
    raw_text: str = await llms.intent_aggregator_llm.ainvoke(prompt)

    final_intents: list[IntentName] = parse_aggregated_intents(raw_text)

    if IntentName.GENERAL not in final_intents:
        final_intents = [IntentName.GENERAL] + final_intents

    sql_needed: bool = any(
        intent in final_intents
        for intent in (
            IntentName.PRIMARY_PARAMETER,
            IntentName.SECONDARY_PARAMETER,
            IntentName.PLAN_ASSessor if False else IntentName.PLAN_ASSESSOR,
        )
    )

    secondary_needed: bool = IntentName.SECONDARY_PARAMETER in final_intents
    plan_assessor_needed: bool = IntentName.PLAN_ASSESSOR in final_intents

    return {
        "final_intents": final_intents,
        "sql_needed": sql_needed,
        "secondary_needed": secondary_needed,
        "plan_assessor_needed": plan_assessor_needed,
        "sql_review_iteration": 0,
        "max_sql_review_iterations": 5,
    }


async def rag_retrieval_node(state: GraphState) -> GraphState:
    question: str = state["user_question"]
    chunks: list[RetrievedChunk] = await retrieve_rag_chunks(question, top_k=8)
    return {"retrieved_chunks": chunks}


async def filter_rag_node(state: GraphState) -> GraphState:
    chunks: list[RetrievedChunk] = state.get("retrieved_chunks", [])
    filtered: list[RetrievedChunk] = filter_chunks_by_score(chunks, threshold=0.70)

    references: list[str] = [chunk.source_document for chunk in filtered]

    return {
        "filtered_chunks": filtered,
        "references": references,
    }


async def sql_generation_node(state: GraphState, llms: LLMRegistry) -> GraphState:
    question: str = state["user_question"]
    final_intents: list[IntentName] = state.get("final_intents", [])

    prompt: str = build_sql_generation_prompt(question, final_intents)
    raw_text: str = await llms.sql_generator_llm.ainvoke(prompt)
    sql_draft: SqlDraft = parse_sql_draft(raw_text)

    return {"sql_draft": sql_draft}


async def sql_critic_node(state: GraphState, llms: LLMRegistry) -> GraphState:
    question: str = state["user_question"]
    sql_draft: Optional[SqlDraft] = state.get("sql_draft")

    if sql_draft is None:
        return {}

    prompt: str = build_sql_critic_prompt(question, sql_draft)
    raw_text: str = await llms.sql_critic_llm.ainvoke(prompt)
    critique: SqlCritique = parse_sql_critique(raw_text)

    next_iteration: int = state.get("sql_review_iteration", 0) + 1

    updated_sql_draft: Optional[SqlDraft] = sql_draft
    if critique.status == SqlReviewStatus.NEEDS_REVISION and critique.suggested_sql is not None:
        updated_sql_draft = SqlDraft(
            sql=critique.suggested_sql,
            explanation=sql_draft.explanation,
            target_database=sql_draft.target_database,
        )

    return {
        "sql_critique": critique,
        "sql_draft": updated_sql_draft,
        "sql_review_iteration": next_iteration,
    }


async def sql_finalize_node(state: GraphState) -> GraphState:
    sql_draft: Optional[SqlDraft] = state.get("sql_draft")
    if sql_draft is None:
        return {}
    return {"sql_final": sql_draft.sql}


async def sql_execute_node(state: GraphState) -> GraphState:
    sql_final: Optional[str] = state.get("sql_final")
    if sql_final is None:
        return {}

    rows: list[QueryResultRow] = await execute_sql_query(sql_final)
    summary: str = summarize_primary_rows(rows)

    return {
        "primary_rows": rows,
        "primary_result_summary": summary,
    }


async def secondary_calculation_node(state: GraphState) -> GraphState:
    question: str = state["user_question"]
    intents: list[IntentName] = state.get("final_intents", [])
    rows: list[QueryResultRow] = state.get("primary_rows", [])

    metric_name: Optional[str] = detect_secondary_metric(question, intents)
    if metric_name is None:
        return {}

    result: SecondaryComputationResult = compute_secondary_metric(metric_name, rows)

    return {
        "secondary_metric_name": metric_name,
        "secondary_result": result,
    }


async def plan_assessor_node(state: GraphState) -> GraphState:
    question: str = state["user_question"]
    rows: list[QueryResultRow] = state.get("primary_rows", [])
    secondary_result: Optional[SecondaryComputationResult] = state.get("secondary_result")

    result: PlanAssessmentResult = assess_plan_feasibility(
        user_question=question,
        rows=rows,
        secondary_result=secondary_result,
    )

    return {"plan_assessment_result": result}


async def final_answer_node(state: GraphState, llms: LLMRegistry) -> GraphState:
    question: str = state["user_question"]
    filtered_chunks: list[RetrievedChunk] = state.get("filtered_chunks", [])
    primary_rows: list[QueryResultRow] = state.get("primary_rows", [])
    secondary_result: Optional[SecondaryComputationResult] = state.get("secondary_result")
    plan_assessment_result: Optional[PlanAssessmentResult] = state.get("plan_assessment_result")

    prompt: str = build_final_answer_prompt(
        user_question=question,
        filtered_chunks=filtered_chunks,
        primary_rows=primary_rows,
        secondary_result=secondary_result,
        plan_assessment_result=plan_assessment_result,
    )
    raw_text: str = await llms.final_answer_llm.ainvoke(prompt)
    answer: FinalAnswer = parse_final_answer(raw_text)

    return {"final_answer": answer}


# ============================================================
# Routing functions
# ============================================================

def route_after_aggregate(state: GraphState) -> list[str]:
    destinations: list[str] = ["rag_retrieval"]

    if state.get("sql_needed", False):
        destinations.append("sql_generation")

    return destinations


def route_after_sql_critic(state: GraphState) -> str:
    critique: Optional[SqlCritique] = state.get("sql_critique")
    current_iteration: int = state.get("sql_review_iteration", 0)
    max_iterations: int = state.get("max_sql_review_iterations", 5)

    if critique is None:
        return "sql_finalize"

    if critique.status == SqlReviewStatus.APPROVED:
        return "sql_finalize"

    if current_iteration >= max_iterations:
        return "sql_finalize"

    return "sql_generation"


def route_after_sql_execute(state: GraphState) -> str:
    if state.get("secondary_needed", False):
        return "secondary_calculation"

    if state.get("plan_assessor_needed", False):
        return "plan_assessor"

    return "final_answer"


def route_after_secondary_calculation(state: GraphState) -> str:
    if state.get("plan_assessor_needed", False):
        return "plan_assessor"
    return "final_answer"


# ============================================================
# Graph builder
# ============================================================

def build_graph(llms: LLMRegistry) -> Any:
    builder: StateGraph[GraphState] = StateGraph(GraphState)

    # ---- intent fan-out
    builder.add_node("deterministic_intent", deterministic_intent_node)
    builder.add_node("llm_intent", lambda state: llm_intent_node(state, llms))
    builder.add_node("semantic_intent", semantic_intent_node)

    # ---- intent aggregation
    builder.add_node("aggregate_intents", lambda state: aggregate_intents_node(state, llms))

    # ---- rag branch
    builder.add_node("rag_retrieval", rag_retrieval_node)
    builder.add_node("filter_rag", filter_rag_node)

    # ---- sql branch
    builder.add_node("sql_generation", lambda state: sql_generation_node(state, llms))
    builder.add_node("sql_critic", lambda state: sql_critic_node(state, llms))
    builder.add_node("sql_finalize", sql_finalize_node)
    builder.add_node("sql_execute", sql_execute_node)

    # ---- post-sql processing
    builder.add_node("secondary_calculation", secondary_calculation_node)
    builder.add_node("plan_assessor", plan_assessor_node)

    # ---- final synthesis
    builder.add_node("final_answer", lambda state: final_answer_node(state, llms))

    # ---- start -> parallel intent detectors
    builder.add_edge(START, "deterministic_intent")
    builder.add_edge(START, "llm_intent")
    builder.add_edge(START, "semantic_intent")

    # ---- fan-in intent detectors -> aggregate
    builder.add_edge("deterministic_intent", "aggregate_intents")
    builder.add_edge("llm_intent", "aggregate_intents")
    builder.add_edge("semantic_intent", "aggregate_intents")

    # ---- aggregate -> fan-out into rag and optional sql
    builder.add_conditional_edges(
        "aggregate_intents",
        route_after_aggregate,
    )

    # ---- rag branch
    builder.add_edge("rag_retrieval", "filter_rag")
    builder.add_edge("filter_rag", "final_answer")

    # ---- sql review loop
    builder.add_edge("sql_generation", "sql_critic")
    builder.add_conditional_edges(
        "sql_critic",
        route_after_sql_critic,
    )
    builder.add_edge("sql_finalize", "sql_execute")

    # ---- sql execute -> conditional post-processing
    builder.add_conditional_edges(
        "sql_execute",
        route_after_sql_execute,
    )
    builder.add_conditional_edges(
        "secondary_calculation",
        route_after_secondary_calculation,
    )
    builder.add_edge("plan_assessor", "final_answer")

    # ---- final
    builder.add_edge("final_answer", END)

    graph: Any = builder.compile()
    return graph