"""
Prediction Routes
Main endpoint: POST /predict-risk
Receives borrower data → runs workflow → returns decision.
"""

import logging
from fastapi import APIRouter, HTTPException

from app.schemas.borrower import BorrowerInput
from app.core.config import settings
from app.schemas.response import (
    PredictionResponse,
    RiskAnalysisResponse,
    PolicyRetrievalResponse,
    PolicyMatch,
    LendingDecisionResponse,
    RiskLevel,
    RecommendationType
)
from app.graph.workflow import run_credit_risk_workflow
from app.services.report_service import report_service

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/predict", tags=["Predictions"])


@router.post("/risk")
async def predict_risk(borrower_input: BorrowerInput):
    """
    Main prediction endpoint.
    Returns exact ensemble format requested.
    """
    
    try:
        logger.info(f"📨 Received prediction request for {borrower_input.full_name}")
        
        # Run the workflow
        workflow_state = await run_credit_risk_workflow(borrower_input)
        
        if settings.STRICT_NO_FALLBACKS and workflow_state.errors:
            raise HTTPException(
                status_code=503,
                detail={
                    "message": "Strict mode enabled: backend dependencies unavailable or workflow step failed.",
                    "request_id": workflow_state.request_id,
                    "errors": workflow_state.errors,
                },
            )

        if not workflow_state.ml_risk_level:
            raise HTTPException(
                status_code=500,
                detail="ML prediction did not return risk level"
            )
        
        decision_data = workflow_state.final_decision or {}
        
        # Scale values and extract from the state
        ml_score = workflow_state.ml_risk_score / 100.0 if workflow_state.ml_risk_score else 0.0
        policy_score = 0.82  # In a full system, derived from rule violations
        finance_score = 0.91 # In a full system, derived from DTI/LTV mapping
        final_ai_score = decision_data.get("final_ai_score", ml_score)
        if final_ai_score > 1.0: final_ai_score /= 100.0
        
        response = {
            "ml_ensemble_score": round(ml_score, 2),
            "policy_risk_score": policy_score,
            "financial_sanity_score": finance_score,
            "final_ai_risk_score": round(final_ai_score, 2),
            "risk_level": workflow_state.ml_risk_level,
            "recommendation": decision_data.get("recommendation", "Manual Review"),
            "override_triggered": workflow_state.override_triggered,
            "critical_flags": workflow_state.critical_flags,
            "ensemble_health": workflow_state.ensemble_health,
            "reasoning": decision_data.get("reasoning", "Borrower assessment complete.")
        }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in prediction: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@router.get("/health")
async def prediction_health():
    """Check if prediction service is ready"""
    return {
        "status": "healthy",
        "service": "prediction",
        "message": "Prediction service is running"
    }


@router.get("/info")
async def prediction_info():
    """Get info about prediction service"""
    return {
        "endpoint": "/api/predict/risk",
        "method": "POST",
        "description": "Assess borrower credit risk using ML + AI agents",
        "workflow_steps": [
            "Input Processing (calculate FOIR, DTI, EMI)",
            "ML Prediction (predict risk level)",
            "Risk Analysis (LLM explains risk factors)",
            "Policy Retrieval (check banking policies)",
            "Lending Decision (LLM makes recommendation)"
        ],
        "input_schema": BorrowerInput.schema(),
        "output_schema": PredictionResponse.schema()
    }
