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
        
        response = {
            "model_scores": workflow_state.ml_model_scores,
            "ensemble_score": workflow_state.ml_risk_score,
            "risk_level": workflow_state.ml_risk_level,
            "disagreement_flag": workflow_state.disagreement_flag,
            "recommendation": decision_data.get("recommendation", "Manual Review"),
            "reasoning": decision_data.get("reasoning", "")
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
