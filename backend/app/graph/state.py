"""
LangGraph State Definition
Defines the state that flows through the multi-agent workflow.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from app.schemas.borrower import BorrowerInput, BorrowerProfile
from app.schemas.response import RiskLevel, RecommendationType


@dataclass
class WorkflowState:
    """
    Complete state that flows through the LangGraph workflow.
    Each agent reads from and writes to this state
    """
    
    # ========== INPUT ==========
    # Original borrower input from frontend
    borrower_input: Optional[BorrowerInput] = None
    
    # ========== CALCULATED METRICS ==========
    # Financial metrics calculated in the first step
    monthly_income: float = 0.0
    existing_emi_monthly: float = 0.0
    loan_amount_requested: float = 0.0
    loan_tenure_months: int = 60
    
    # Calculated ratios
    foir: float = 0.0  # Fixed Obligation to Income Ratio
    dti: float = 0.0   # Debt-to-Income Ratio
    proposed_emi: float = 0.0  # Proposed EMI for new loan
    total_emi_after_loan: float = 0.0
    
    # ========== ML PREDICTION ==========
    # From ML Service
    ml_risk_level: Optional[str] = None
    ml_risk_score: float = 0.0
    disagreement_flag: str = "Low"
    ml_model_scores: Dict[str, Any] = field(default_factory=dict)
    prediction_confidence: float = 0.95
    fallback_level_used: int = 1
    ensemble_health: str = "healthy"
    override_triggered: bool = False
    critical_flags: List[str] = field(default_factory=list)
    score_breakdown: Dict[str, Any] = field(default_factory=dict)
    
    # ========== RISK ANALYSIS AGENT OUTPUT ==========
    # From Risk Analysis Agent (LLM)
    risk_analysis: Dict[str, Any] = field(default_factory=dict)
    # Expected keys:
    # - top_risk_factors: List[str]
    # - positive_factors: List[str]
    # - confidence_score: float
    
    # ========== POLICY RETRIEVAL AGENT OUTPUT ==========
    # From Policy Retrieval Agent (RAG)
    policy_matches: List[Dict[str, str]] = field(default_factory=list)
    # Each item:
    # {
    #   "rule_name": str,
    #   "rule_text": str,
    #   "status": "Compliant" | "Borderline" | "Violated"
    # }
    
    policy_violations: int = 0
    policy_compliances: int = 0
    
    # ========== AI SCORING AGENT OUTPUT ==========
    # From AI Scoring Agent (LLM)
    final_ai_score: float = 0.0
    ai_score_reasoning: str = ""
    final_risk_level: Optional[RiskLevel] = None
    
    # ========== DECISION AGENT OUTPUT ==========
    # From Lending Decision Agent (LLM)
    final_decision: Dict[str, Any] = field(default_factory=dict)
    # Expected keys:
    # - recommendation: str
    # - reasoning: str
    
    # ========== TRACKING ==========
    # For debugging and monitoring
    request_id: str = ""
    errors: List[str] = field(default_factory=list)
    step_completed: str = ""
    agent_interactions: List[Dict[str, Any]] = field(default_factory=list)
    workflow_trace: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_error(self, error_message: str):
        """Add an error message to the state"""
        self.errors.append(error_message)
    
    def to_dict(self) -> dict:
        """Convert state to dictionary (for serialization)"""
        return {
            "request_id": self.request_id,
            "borrower_input": self.borrower_input.dict() if self.borrower_input else None,
            "foir": self.foir,
            "dti": self.dti,
            "proposed_emi": self.proposed_emi,
            "ml_risk_level": self.ml_risk_level,
            "ml_risk_score": self.ml_risk_score,
            "disagreement_flag": self.disagreement_flag,
            "ml_model_scores": self.ml_model_scores,
            "prediction_confidence": self.prediction_confidence,
            "fallback_level_used": self.fallback_level_used,
            "ensemble_health": self.ensemble_health,
            "override_triggered": self.override_triggered,
            "critical_flags": self.critical_flags,
            "score_breakdown": self.score_breakdown,
            "risk_analysis": self.risk_analysis,
            "policy_matches": self.policy_matches,
            "final_ai_score": self.final_ai_score,
            "ai_score_reasoning": self.ai_score_reasoning,
            "final_risk_level": self.final_risk_level.value if self.final_risk_level else None,
            "final_decision": self.final_decision,
            "agent_interactions": self.agent_interactions,
            "workflow_trace": self.workflow_trace,
            "errors": self.errors
        }


def create_empty_state() -> WorkflowState:
    """Create a new empty state"""
    return WorkflowState()
