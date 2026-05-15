"""
LangGraph Workflow Orchestration
Defines the multi-agent workflow for credit risk assessment.

Workflow flow:
Input → ML Prediction → Risk Analysis → Policy Retrieval → Decision → Output
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
from app.services.groq_service import groq_service, risk_agent, decision_agent, scoring_agent
from app.services.rag_service import rag_service
from app.services.report_service import report_service

logger = logging.getLogger(__name__)


def _normalize_workflow_state(raw_state: Any) -> WorkflowState:
    """Normalize workflow output to WorkflowState for compatibility."""
    if isinstance(raw_state, WorkflowState):
        return raw_state

    if isinstance(raw_state, dict):
        normalized = WorkflowState()
        for field_name in WorkflowState.__dataclass_fields__.keys():
            if field_name in raw_state:
                setattr(normalized, field_name, raw_state[field_name])
        return normalized

    raise TypeError(f"Unexpected workflow state type: {type(raw_state)}")


def _append_trace(
    state: WorkflowState,
    *,
    step: str,
    node: str,
    status: str,
    started_at: datetime,
    input_data: Dict[str, Any],
    output_data: Dict[str, Any],
    model: str,
    source: str,
    note: str = "",
) -> None:
    """Append one normalized workflow trace event for frontend timeline rendering."""
    ended_at = datetime.utcnow()
    duration_ms = max(0, int((ended_at - started_at).total_seconds() * 1000))
    state.workflow_trace.append(
        {
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
        }
    )


# ============================================================================
# NODE: INPUT PROCESSING & CALCULATIONS
# ============================================================================

def node_input_processing(state: WorkflowState) -> WorkflowState:
    """
    Process input and calculate financial metrics.
    
    Calculates:
    - FOIR (Fixed Obligation to Income Ratio)
    - DTI (Debt-to-Income Ratio)
    - Proposed EMI
    """
    
    logger.info("📥 Node: Input Processing")
    started_at = datetime.utcnow()
    
    try:
        borrower = state.borrower_input
        
        # Extract values
        monthly_income = borrower.monthly_income
        existing_emi = borrower.existing_emi_monthly
        loan_amount = borrower.loan_amount_requested
        tenure_months = borrower.loan_tenure_months
        existing_loans = borrower.existing_loan_amount
        
        # Store in state
        state.monthly_income = monthly_income
        state.existing_emi_monthly = existing_emi
        state.loan_amount_requested = loan_amount
        state.loan_tenure_months = tenure_months
        
        # ========== CALCULATE PROPOSED EMI ==========
        # Formula: EMI = P * [r(1+r)^n] / [(1+r)^n - 1]
        # Where: P = Principal, r = monthly rate, n = tenure in months
        # Simplified: Using 10% annual rate as standard
        
        annual_rate = 0.10  # 10% per annum
        monthly_rate = annual_rate / 12
        
        if monthly_rate > 0:
            numerator = loan_amount * monthly_rate * ((1 + monthly_rate) ** tenure_months)
            denominator = ((1 + monthly_rate) ** tenure_months) - 1
            proposed_emi = numerator / denominator
        else:
            proposed_emi = loan_amount / tenure_months
        
        state.proposed_emi = round(proposed_emi, 2)
        
        logger.info(f"   Proposed EMI: ₹{state.proposed_emi:,.0f}")
        
        # ========== CALCULATE FOIR (Fixed Obligation to Income Ratio) ==========
        # FOIR = (Existing EMI + Proposed EMI) / Monthly Income
        # Banks prefer FOIR < 45%
        
        total_emi = existing_emi + proposed_emi
        state.total_emi_after_loan = round(total_emi, 2)
        
        if monthly_income > 0:
            foir = (total_emi / monthly_income) * 100
            state.foir = round(foir / 100, 4)  # Store as decimal (0.45 = 45%)
        else:
            state.foir = 0.0
        
        logger.info(f"   FOIR: {state.foir*100:.2f}%")
        
        total_debt = existing_loans + loan_amount
        
        if monthly_income > 0:
            # Standard DTI calculation: Total Debt / Annual Income
            annual_income = monthly_income * 12
            dti = (total_debt / annual_income) * 100
            state.dti = round(dti / 100, 4)  # Store as decimal
        else:
            state.dti = 0.0
        
        logger.info(f"   DTI: {state.dti*100:.2f}%")
        
        # Create enriched borrower profile
        borrower_profile = BorrowerProfile(
            full_name=borrower.full_name,
            age=borrower.age,
            monthly_income=monthly_income,
            employment_type=borrower.employment_type,
            credit_score=borrower.credit_score,
            existing_loan_amount=existing_loans,
            existing_emi_monthly=existing_emi,
            loan_amount_requested=loan_amount,
            loan_purpose=borrower.loan_purpose,
            loan_tenure_months=tenure_months,
            foir=state.foir,
            dti=state.dti,
            proposed_emi=state.proposed_emi,
            total_emi_after_loan=state.total_emi_after_loan,
        )
        
        # Store in state (we'll need this for ML service)
        state.step_completed = "input_processing"
        logger.info("✓ Input processing complete")

        _append_trace(
            state,
            step="input_processing",
            node="Input Processing",
            status="completed",
            started_at=started_at,
            model="financial_formula_engine_v1",
            source="backend",
            note="EMI, FOIR, and DTI computed from borrower input.",
            input_data={
                "monthly_income": monthly_income,
                "existing_emi_monthly": existing_emi,
                "loan_amount_requested": loan_amount,
                "loan_tenure_months": tenure_months,
                "existing_loan_amount": existing_loans,
                "annual_rate": annual_rate,
            },
            output_data={
                "proposed_emi": state.proposed_emi,
                "total_emi_after_loan": state.total_emi_after_loan,
                "foir": state.foir,
                "dti": state.dti,
            },
        )
        
        return state
        
    except Exception as e:
        logger.error(f"❌ Error in input processing: {str(e)}")
        state.add_error(f"Input processing failed: {str(e)}")
        state.step_completed = "input_processing_failed"
        _append_trace(
            state,
            step="input_processing",
            node="Input Processing",
            status="failed",
            started_at=started_at,
            model="financial_formula_engine_v1",
            source="backend",
            note=str(e),
            input_data={},
            output_data={"error": str(e)},
        )
        return state


# ============================================================================
# NODE: ML PREDICTION
# ============================================================================

def node_ml_prediction(state: WorkflowState) -> WorkflowState:
    """
    Call ML service to predict credit risk.
    """
    
    logger.info("🤖 Node: ML Prediction")
    started_at = datetime.utcnow()
    
    try:
        borrower = state.borrower_input
        
        # Create profile for ML service
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
        
        # Get ML prediction
        prediction_result = ml_service.predict_all_models(borrower_profile)
        
        # Store in state
        state.ml_risk_level = prediction_result.get("risk_level")
        state.ml_risk_score = prediction_result.get("ml_ensemble_score", 0.0) * 100
        state.disagreement_flag = prediction_result.get("disagreement_flag")
        state.ml_model_scores = prediction_result.get("models", {})
        
        # New robust architecture telemetry
        state.prediction_confidence = prediction_result.get("prediction_confidence", 0.95)
        state.fallback_level_used = prediction_result.get("fallback_level_used", 1)
        state.ensemble_health = prediction_result.get("ensemble_health", "healthy")
        state.override_triggered = prediction_result.get("override_triggered", False)
        state.critical_flags = prediction_result.get("critical_flags", [])
        
        logger.info(f"   Risk Level: {state.ml_risk_level}")
        logger.info(f"   Risk Score: {state.ml_risk_score}")
        logger.info(f"   Disagreement: {state.disagreement_flag}")
        logger.info(f"   Ensemble Health: {state.ensemble_health} (Fallback Level: {state.fallback_level_used})")
        
        state.step_completed = "ml_prediction"
        logger.info("✓ ML prediction complete")

        _append_trace(
            state,
            step="ml_prediction",
            node="ML Prediction",
            status="completed",
            started_at=started_at,
            model="ensemble",
            source="ml_service",
            note="Risk score computed by ML ensemble.",
            input_data={
                "credit_score": borrower.credit_score,
                "employment_type": borrower.employment_type,
                "foir": state.foir,
                "dti": state.dti,
                "loan_amount_requested": state.loan_amount_requested,
                "monthly_income": state.monthly_income,
            },
            output_data=prediction_result,
        )
        
        return state
        
    except Exception as e:
        logger.error(f"❌ Error in ML prediction: {str(e)}")
        state.add_error(f"ML prediction failed: {str(e)}")
        state.step_completed = "ml_prediction_failed"
        _append_trace(
            state,
            step="ml_prediction",
            node="ML Prediction",
            status="failed",
            started_at=started_at,
            model=settings.ML_MODEL_PATH,
            source="ml_service",
            note=str(e),
            input_data={},
            output_data={"error": str(e)},
        )
        return state


# ============================================================================
# NODE: RISK ANALYSIS AGENT
# ============================================================================

def node_risk_analysis(state: WorkflowState) -> WorkflowState:
    """
    Use LLM to analyze risk factors.
    Real Groq API call (no mocks).
    """
    
    logger.info("🔍 Node: Risk Analysis Agent")
    started_at = datetime.utcnow()
    
    try:
        borrower = state.borrower_input
        
        # Build borrower data dict for LLM
        borrower_data = {
            "age": borrower.age,
            "credit_score": borrower.credit_score,
            "monthly_income": state.monthly_income,
            "employment_type": borrower.employment_type,
            "foir": state.foir,
            "dti": state.dti,
            "loan_amount_requested": state.loan_amount_requested,
            "existing_loans": borrower.existing_loan_amount
        }
        
        # Call the risk analysis agent
        analysis, interaction = risk_agent.analyze(
            borrower_data=borrower_data,
            risk_score=state.ml_risk_score,
            risk_level=state.ml_risk_level if state.ml_risk_level else "Unknown"
        )
        
        state.risk_analysis = analysis
        state.agent_interactions.append(interaction)
        
        logger.info(f"   Top Risk Factors: {analysis.get('top_risk_factors', [])}")
        logger.info(f"   Positive Factors: {analysis.get('positive_factors', [])}")
        
        state.step_completed = "risk_analysis"
        logger.info("✓ Risk analysis complete")

        _append_trace(
            state,
            step="risk_analysis",
            node="Risk Analysis Agent",
            status="completed",
            started_at=started_at,
            model=settings.GROQ_MODEL,
            source=str(analysis.get("agent_source", "unknown")),
            note="LLM analyzed risk factors from borrower and ML context.",
            input_data=borrower_data,
            output_data={
                "top_risk_factors": analysis.get("top_risk_factors", []),
                "positive_factors": analysis.get("positive_factors", []),
                "confidence_score": analysis.get("confidence_score"),
                "warnings": analysis.get("warnings", []),
            },
        )
        
        return state
        
    except Exception as e:
        logger.error(f"❌ Error in risk analysis: {str(e)}")
        state.add_error(f"Risk analysis failed: {str(e)}")
        state.step_completed = "risk_analysis_failed"
        _append_trace(
            state,
            step="risk_analysis",
            node="Risk Analysis Agent",
            status="failed",
            started_at=started_at,
            model=settings.GROQ_MODEL,
            source="groq",
            note=str(e),
            input_data={},
            output_data={"error": str(e)},
        )
        return state


# ============================================================================
# NODE: POLICY RETRIEVAL
# ============================================================================

def node_policy_retrieval(state: WorkflowState) -> WorkflowState:
    """
    Retrieve applicable policies from knowledge base using RAG.
    """
    
    logger.info("📋 Node: Policy Retrieval")
    started_at = datetime.utcnow()
    
    try:
        borrower = state.borrower_input
        borrower_context = {
            "age": borrower.age,
            "monthly_income": state.monthly_income,
            "employment_type": borrower.employment_type,
            "credit_score": borrower.credit_score,
            "loan_amount_requested": state.loan_amount_requested,
            "loan_purpose": borrower.loan_purpose,
            "foir": state.foir,
            "dti": state.dti,
        }

        retrieval_result = rag_service.retrieve_policies(borrower_context)
        policies = retrieval_result.get("policies", [])

        violations = sum(1 for p in policies if p.get("status") == "Violated")
        compliances = sum(1 for p in policies if p.get("status") == "Compliant")

        state.policy_matches = policies
        state.policy_violations = violations
        state.policy_compliances = compliances

        # Log policy retrieval interaction
        state.agent_interactions.append({
            "agent": "Policy Retrieval Agent",
            "prompt": f"Retrieving policies for borrower with credit score {borrower.credit_score}, FOIR {state.foir*100:.1f}%, and loan amount {state.loan_amount_requested}.",
            "response": f"Retrieved {len(policies)} matching policies. Detected {violations} violations and {compliances} compliances.",
            "metadata": {
                "violations": violations,
                "compliances": compliances,
                "rules_checked": len(policies)
            }
        })

        for warning in retrieval_result.get("warnings", []):
            state.add_error(warning)
        
        logger.info(f"   Policies Checked: {len(policies)}")
        logger.info(f"   Violations: {violations}, Compliances: {compliances}")
        
        state.step_completed = "policy_retrieval"
        logger.info("✓ Policy retrieval complete")

        _append_trace(
            state,
            step="policy_retrieval",
            node="Policy Retrieval Agent",
            status="completed",
            started_at=started_at,
            model="qdrant_lexical_retrieval",
            source=str(retrieval_result.get("source", "unknown")),
            note="Retrieved and classified policies against borrower metrics.",
            input_data=borrower_context,
            output_data={
                "rules_checked": len(policies),
                "violations": violations,
                "compliances": compliances,
                "policies": policies,
                "warnings": retrieval_result.get("warnings", []),
            },
        )
        
        return state
        
    except Exception as e:
        logger.error(f"❌ Error in policy retrieval: {str(e)}")
        state.add_error(f"Policy retrieval failed: {str(e)}")
        state.step_completed = "policy_retrieval_failed"
        _append_trace(
            state,
            step="policy_retrieval",
            node="Policy Retrieval Agent",
            status="failed",
            started_at=started_at,
            model="qdrant_lexical_retrieval",
            source="rag_service",
            note=str(e),
            input_data={},
            output_data={"error": str(e)},
        )
        return state


# ============================================================================
# NODE: AI SCORING AGENT
# ============================================================================

def node_ai_scoring(state: WorkflowState) -> WorkflowState:
    """
    Synthesize final score using AI agent.
    """
    
    logger.info("⚖️ Node: AI Scoring Agent")
    started_at = datetime.utcnow()
    
    try:
        borrower = state.borrower_input
        borrower_data = {
            "foir": state.foir,
            "dti": state.dti,
            "credit_score": borrower.credit_score,
            "employment_type": borrower.employment_type
        }
        
        result, interaction = scoring_agent.evaluate(
            ml_score=state.ml_risk_score,
            ml_level=state.ml_risk_level.value if hasattr(state.ml_risk_level, 'value') else str(state.ml_risk_level),
            prediction_confidence=state.prediction_confidence,
            fallback_level_used=state.fallback_level_used,
            ensemble_health=state.ensemble_health,
            override_triggered=state.override_triggered,
            critical_flags=state.critical_flags,
            risk_analysis=state.risk_analysis,
            policy_matches=state.policy_matches,
            borrower_data=borrower_data,
            model_scores=state.ml_model_scores
        )
        
        state.final_ai_score = round(float(result.get("final_score", state.ml_risk_score)), 2)
        state.ai_score_reasoning = result.get("reasoning", "")
        state.agent_interactions.append(interaction)
        
        # Calculate final risk level based on AI score
        if state.final_ai_score < 30:
            state.final_risk_level = RiskLevel.LOW
        elif state.final_ai_score < 60:
            state.final_risk_level = RiskLevel.MEDIUM
        else:
            state.final_risk_level = RiskLevel.HIGH
            
        logger.info(f"   Final AI Score: {state.final_ai_score}")
        logger.info(f"   Final Risk Level: {state.final_risk_level.value}")
        
        state.step_completed = "ai_scoring"
        logger.info("✓ AI scoring complete")

        _append_trace(
            state,
            step="ai_scoring",
            node="AI Scoring Agent",
            status="completed",
            started_at=started_at,
            model=settings.GROQ_MODEL,
            source="groq",
            note="AI synthesis of ML score and company policies.",
            input_data={
                "ml_score": state.ml_risk_score,
                "policies_count": len(state.policy_matches)
            },
            output_data={
                "final_ai_score": state.final_ai_score,
                "reasoning": state.ai_score_reasoning
            },
        )
        
        return state
        
    except Exception as e:
        logger.error(f"❌ Error in AI scoring: {str(e)}")
        state.add_error(f"AI scoring failed: {str(e)}")
        state.step_completed = "ai_scoring_failed"
        _append_trace(
            state,
            step="ai_scoring",
            node="AI Scoring Agent",
            status="failed",
            started_at=started_at,
            model=settings.GROQ_MODEL,
            source="groq",
            note=str(e),
            input_data={},
            output_data={"error": str(e)},
        )
        return state


# ============================================================================
# NODE: LENDING DECISION
# ============================================================================

def node_lending_decision(state: WorkflowState) -> WorkflowState:
    """
    Use LLM to make final lending decision.
    Real Groq API call (no mocks).
    """
    
    logger.info("⚖️ Node: Lending Decision Agent")
    started_at = datetime.utcnow()
    
    try:
        borrower = state.borrower_input
        
        # Build borrower data dict for decision agent
        borrower_data = {
            "age": borrower.age,
            "credit_score": borrower.credit_score,
            "monthly_income": state.monthly_income,
            "employment_type": borrower.employment_type,
            "foir": state.foir,
            "dti": state.dti,
            "loan_amount_requested": state.loan_amount_requested,
            "existing_loans": borrower.existing_loan_amount
        }
        
        # Call decision agent
        decision, interaction = decision_agent.decide(
            model_scores=state.ml_model_scores,
            ml_risk_level=state.ml_risk_level.value if hasattr(state.ml_risk_level, 'value') else str(state.ml_risk_level),
            ml_risk_score=state.ml_risk_score,
            disagreement_flag=state.disagreement_flag,
            prediction_confidence=state.prediction_confidence,
            fallback_level_used=state.fallback_level_used,
            ensemble_health=state.ensemble_health,
            override_triggered=state.override_triggered,
            critical_flags=state.critical_flags,
            policy_matches=state.policy_matches,
            borrower_data=borrower_data
        )
        
        state.final_decision = decision
        state.agent_interactions.append(interaction)
        
        logger.info(f"   Recommendation: {decision.get('recommendation')}")
        logger.info(f"   Reasoning: {decision.get('reasoning')}")
        
        state.step_completed = "lending_decision"
        logger.info("✓ Lending decision complete")

        _append_trace(
            state,
            step="lending_decision",
            node="Lending Decision Agent",
            status="completed",
            started_at=started_at,
            model=settings.GROQ_MODEL,
            source=str(decision.get("agent_source", "unknown")),
            note="Final recommendation generated from risk and policy context.",
            input_data={
                "risk_analysis": {
                    **state.risk_analysis,
                    "ai_refined_score": state.final_ai_score,
                    "ai_reasoning": state.ai_score_reasoning
                },
                "policy_matches": state.policy_matches,
                "borrower_context": borrower_data,
            },
            output_data={
                "recommendation": decision.get("recommendation"),
                "reasoning": decision.get("reasoning"),
                "warnings": decision.get("warnings", []),
            },
        )
        
        return state
        
    except Exception as e:
        logger.error(f"❌ Error in lending decision: {str(e)}")
        state.add_error(f"Lending decision failed: {str(e)}")
        state.step_completed = "lending_decision_failed"
        _append_trace(
            state,
            step="lending_decision",
            node="Lending Decision Agent",
            status="failed",
            started_at=started_at,
            model=settings.GROQ_MODEL,
            source="groq",
            note=str(e),
            input_data={},
            output_data={"error": str(e)},
        )
        return state


# ============================================================================
# NODE: REPORT AGENT
# ============================================================================

def node_report_agent(state: WorkflowState) -> WorkflowState:
    """
    Save the final report snapshot.
    """
    logger.info("📝 Node: Report Agent")
    
    # Save the report using the report service
    report_data = report_service.build_report(state)
    report_service.save_report(report_data)
    
    state.step_completed = "report_agent"
    return state


# ============================================================================
# BUILD THE WORKFLOW GRAPH
# ============================================================================

def build_workflow():
    """
    Build and compile the LangGraph workflow.
    """
    
    # Create graph
    workflow = StateGraph(WorkflowState)
    
    # Add nodes (functions)
    workflow.add_node("input_processing", node_input_processing)
    workflow.add_node("ml_prediction", node_ml_prediction)
    workflow.add_node("risk_analysis", node_risk_analysis)
    workflow.add_node("policy_retrieval", node_policy_retrieval)
    workflow.add_node("ai_scoring", node_ai_scoring)
    workflow.add_node("decision_agent", node_lending_decision)
    workflow.add_node("report_agent", node_report_agent)
    
    # Set entry point
    workflow.set_entry_point("input_processing")
    
    # Add edges (flow)
    workflow.add_edge("input_processing", "ml_prediction")
    workflow.add_edge("ml_prediction", "risk_analysis")
    workflow.add_edge("risk_analysis", "policy_retrieval")
    workflow.add_edge("policy_retrieval", "ai_scoring")
    workflow.add_edge("ai_scoring", "decision_agent")
    workflow.add_edge("decision_agent", "report_agent")
    workflow.add_edge("report_agent", END)
    
    # Compile
    graph = workflow.compile()
    
    logger.info("✓ LangGraph workflow compiled successfully")
    
    return graph


# Create global workflow instance
credit_risk_workflow = build_workflow()


# ============================================================================
# RUN WORKFLOW
# ============================================================================

async def run_credit_risk_workflow(borrower_input: BorrowerInput) -> WorkflowState:
    """
    Execute the complete credit risk assessment workflow.
    
    Args:
        borrower_input: BorrowerInput from frontend form
        
    Returns:
        Final WorkflowState with all results
    """
    
    logger.info("=" * 80)
    logger.info("🚀 STARTING CREDIT RISK ASSESSMENT WORKFLOW")
    logger.info("=" * 80)
    
    # Create initial state
    initial_state = WorkflowState(
        borrower_input=borrower_input,
        request_id=f"REQ_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    )
    
    logger.info(f"Request ID: {initial_state.request_id}")
    logger.info(f"Borrower: {borrower_input.full_name}")
    
    # Run workflow
    try:
        final_state_raw = credit_risk_workflow.invoke(initial_state)
        final_state = _normalize_workflow_state(final_state_raw)
        
        logger.info("=" * 80)
        logger.info("✓ WORKFLOW COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
        return final_state
        
    except Exception as e:
        logger.error(f"❌ WORKFLOW FAILED: {str(e)}")
        initial_state.add_error(f"Workflow execution failed: {str(e)}")
        return initial_state
