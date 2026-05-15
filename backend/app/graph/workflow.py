"""
LangGraph Workflow Orchestration
Defines the multi-agent workflow for credit risk assessment.
"""

import logging
import json
import uuid
from datetime import datetime
from typing import Dict, Any

from langgraph.graph import StateGraph, END
from app.graph.state import WorkflowState
from app.schemas.borrower import BorrowerInput, BorrowerProfile
from app.schemas.response import RiskLevel
from app.core.config import settings
from app.services.ml_service import ml_service
from app.services.groq_service import (
    groq_service, risk_agent, decision_agent, 
    policy_agent, arbitration_agent
)
from app.services.rag_service import rag_service
from app.services.report_service import report_service

logger = logging.getLogger(__name__)

def _normalize_workflow_state(raw_state: Any) -> WorkflowState:
    if isinstance(raw_state, WorkflowState):
        return raw_state
    if isinstance(raw_state, dict):
        normalized = WorkflowState()
        for field_name in WorkflowState.__dataclass_fields__.keys():
            if field_name in raw_state:
                setattr(normalized, field_name, raw_state[field_name])
        return normalized
    raise TypeError(f"Unexpected workflow state type: {type(raw_state)}")

def _append_trace(state: WorkflowState, *, step: str, node: str, status: str, started_at: datetime, input_data: Dict[str, Any], output_data: Dict[str, Any], model: str, source: str, note: str = ""):
    ended_at = datetime.utcnow()
    duration_ms = max(0, int((ended_at - started_at).total_seconds() * 1000))
    state.workflow_trace.append({
        "step": step,
        "node": node,
        "status": status,
        "started_at": started_at.isoformat() + "Z",
        "ended_at": ended_at.isoformat() + "Z",
        "duration_ms": duration_ms,
        "model": model,
        "source": source,
        "note": note,
        "input": input_data,
        "output": output_data,
    })

# ============================================================================
# NODES
# ============================================================================

def node_input_processing(state: WorkflowState) -> WorkflowState:
    logger.info("📥 Node: Input Processing")
    started_at = datetime.utcnow()
    try:
        borrower = state.borrower_input
        state.monthly_income = borrower.monthly_income
        state.existing_emi_monthly = borrower.existing_emi_monthly
        state.loan_amount_requested = borrower.loan_amount_requested
        state.loan_tenure_months = borrower.loan_tenure_months
        
        # EMI Calculation (10% rate)
        loan_amount = borrower.loan_amount_requested
        tenure = borrower.loan_tenure_months
        monthly_rate = 0.10 / 12
        numerator = loan_amount * monthly_rate * ((1 + monthly_rate) ** tenure)
        denominator = ((1 + monthly_rate) ** tenure) - 1
        state.proposed_emi = round(numerator / denominator, 2)
        
        state.total_emi_after_loan = round(borrower.existing_emi_monthly + state.proposed_emi, 2)
        state.foir = round(state.total_emi_after_loan / borrower.monthly_income, 4)
        state.dti = round((borrower.existing_loan_amount + loan_amount) / (borrower.monthly_income * 12), 4)
        
        state.step_completed = "input_processing"
        _append_trace(state, step="input_processing", node="Input Processing", status="completed", started_at=started_at, model="formula_v1", source="backend", input_data={}, output_data={"foir": state.foir, "dti": state.dti})
        return state
    except Exception as e:
        state.add_error(str(e))
        return state

def node_ml_prediction(state: WorkflowState) -> WorkflowState:
    logger.info("🤖 Node: ML Prediction")
    started_at = datetime.utcnow()
    try:
        borrower = state.borrower_input
        borrower_profile = BorrowerProfile(
            full_name=borrower.full_name,
            age=borrower.age,
            monthly_income=state.monthly_income,
            employment_type=borrower.employment_type,
            credit_score=borrower.credit_score,
            existing_loan_amount=borrower.existing_loan_amount,
            existing_emi_monthly=state.existing_emi_monthly,
            loan_amount_requested=state.loan_amount_requested,
            loan_purpose=borrower.loan_purpose,
            loan_tenure_months=state.loan_tenure_months,
            foir=state.foir,
            dti=state.dti,
            proposed_emi=state.proposed_emi,
            total_emi_after_loan=state.total_emi_after_loan,
        )
        prediction = ml_service.predict_all_models(borrower_profile)
        state.ml_risk_level = prediction.get("risk_level")
        state.ml_risk_score = prediction.get("ml_ensemble_score", 0.0) * 100
        state.disagreement_score = prediction.get("disagreement_score", 0.0)
        state.prediction_confidence = prediction.get("prediction_confidence", 0.9)
        state.ensemble_health = prediction.get("ensemble_health", "healthy")
        state.override_triggered = prediction.get("override_triggered", False)
        
        state.step_completed = "ml_prediction"
        _append_trace(state, step="ml_prediction", node="ML Prediction", status="completed", started_at=started_at, model="ensemble", source="ml_service", input_data={}, output_data=prediction)
        return state
    except Exception as e:
        state.add_error(str(e))
        return state

def node_risk_analysis(state: WorkflowState) -> WorkflowState:
    logger.info("🔍 Node: Risk Analysis Agent")
    started_at = datetime.utcnow()
    try:
        borrower_data = {"age": state.borrower_input.age, "credit_score": state.borrower_input.credit_score, "foir": state.foir, "dti": state.dti}
        analysis, interaction = risk_agent.analyze(borrower_data, state.ml_risk_score)
        state.risk_analysis = analysis
        state.agent_interactions.append(interaction)
        _append_trace(state, step="risk_analysis", node="Risk Analysis Agent", status="completed", started_at=started_at, model="groq", source="groq", input_data=borrower_data, output_data=analysis)
        return state
    except Exception as e:
        state.add_error(str(e))
        return state

def node_policy_retrieval(state: WorkflowState) -> WorkflowState:
    logger.info("📋 Node: Policy Retrieval")
    started_at = datetime.utcnow()
    try:
        borrower_context = {"credit_score": state.borrower_input.credit_score, "foir": state.foir, "dti": state.dti}
        retrieval = rag_service.retrieve_policies(borrower_context)
        state.policy_matches = retrieval.get("policies", [])
        _append_trace(state, step="policy_retrieval", node="Policy Retrieval", status="completed", started_at=started_at, model="qdrant", source="rag_service", input_data=borrower_context, output_data=retrieval)
        return state
    except Exception as e:
        state.add_error(str(e))
        return state

def node_policy_evaluation(state: WorkflowState) -> WorkflowState:
    logger.info("🛡️ Node: Policy Evaluation Agent")
    started_at = datetime.utcnow()
    try:
        borrower_data = {"credit_score": state.borrower_input.credit_score, "foir": state.foir, "dti": state.dti}
        eval_result, interaction = policy_agent.evaluate(state.policy_matches, borrower_data)
        state.policy_risk_score = eval_result.get("policy_risk_score", 0.0)
        state.agent_interactions.append(interaction)
        _append_trace(state, step="policy_evaluation", node="Policy Evaluation Agent", status="completed", started_at=started_at, model="groq", source="groq", input_data={}, output_data=eval_result)
        return state
    except Exception as e:
        return state

def node_arbitration(state: WorkflowState) -> WorkflowState:
    logger.info("⚖️ Node: Risk Arbitration Agent")
    started_at = datetime.utcnow()
    try:
        ml_data = {"ml_ensemble_score": state.ml_risk_score, "prediction_confidence": state.prediction_confidence, "ensemble_health": state.ensemble_health}
        policy_data = {"policy_risk_score": state.policy_risk_score}
        result, interaction = arbitration_agent.arbitrate(ml_data, state.risk_analysis, policy_data)
        state.final_ai_score = round(float(result.get("final_ai_score", state.ml_risk_score)), 2)
        state.arbitration_summary = result.get("arbitration_summary", "")
        state.agent_interactions.append(interaction)
        
        if state.final_ai_score < 40: state.final_risk_level = RiskLevel.LOW
        elif state.final_ai_score < 65: state.final_risk_level = RiskLevel.MEDIUM
        else: state.final_risk_level = RiskLevel.HIGH
        
        _append_trace(state, step="arbitration", node="Risk Arbitration Agent", status="completed", started_at=started_at, model="groq", source="groq", input_data={}, output_data=result)
        return state
    except Exception as e:
        return state

def node_lending_decision(state: WorkflowState) -> WorkflowState:
    logger.info("⚖️ Node: Lending Decision Agent")
    started_at = datetime.utcnow()
    try:
        decision, interaction = decision_agent.decide({"arbitration_summary": state.arbitration_summary, "final_ai_score": state.final_ai_score}, state.final_risk_level.value if state.final_risk_level else "Unknown")
        state.final_decision = decision
        state.agent_interactions.append(interaction)
        _append_trace(state, step="lending_decision", node="Lending Decision Agent", status="completed", started_at=started_at, model="groq", source="groq", input_data={}, output_data=decision)
        return state
    except Exception as e:
        return state

def node_report_agent(state: WorkflowState) -> WorkflowState:
    logger.info("📝 Node: Report Agent")
    report_data = report_service.build_report(state)
    report_service.save_report(report_data)
    return state

# ============================================================================
# WORKFLOW
# ============================================================================

def build_workflow():
    workflow = StateGraph(WorkflowState)
    workflow.add_node("input_processing", node_input_processing)
    workflow.add_node("ml_prediction", node_ml_prediction)
    workflow.add_node("risk_analysis", node_risk_analysis)
    workflow.add_node("policy_retrieval", node_policy_retrieval)
    workflow.add_node("policy_evaluation", node_policy_evaluation)
    workflow.add_node("arbitration", node_arbitration)
    workflow.add_node("decision_agent", node_lending_decision)
    workflow.add_node("report_agent", node_report_agent)
    
    workflow.set_entry_point("input_processing")
    workflow.add_edge("input_processing", "ml_prediction")
    workflow.add_edge("ml_prediction", "risk_analysis")
    workflow.add_edge("risk_analysis", "policy_retrieval")
    workflow.add_edge("policy_retrieval", "policy_evaluation")
    workflow.add_edge("policy_evaluation", "arbitration")
    workflow.add_edge("arbitration", "decision_agent")
    workflow.add_edge("decision_agent", "report_agent")
    workflow.add_edge("report_agent", END)
    
    return workflow.compile()

credit_risk_workflow = build_workflow()

async def run_credit_risk_workflow(borrower_input: BorrowerInput) -> WorkflowState:
    initial_state = WorkflowState(borrower_input=borrower_input, request_id=f"REQ_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}")
    try:
        final_state_raw = credit_risk_workflow.invoke(initial_state)
        return _normalize_workflow_state(final_state_raw)
    except Exception as e:
        initial_state.add_error(str(e))
        return initial_state
