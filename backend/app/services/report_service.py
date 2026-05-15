"""
Report Service
Builds structured JSON report payloads and caches by request id.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from app.graph.state import WorkflowState


import logging

logger = logging.getLogger(__name__)

class ReportService:
	"""Service for building and retrieving structured borrower reports."""

	def __init__(self) -> None:
		self._cache: Dict[str, Dict[str, Any]] = {}
		logger.info("🆕 ReportService initialized with empty cache")

	def build_report(self, state: WorkflowState) -> Dict[str, Any]:
		"""Build a complete JSON report from workflow state aligned with PredictionResponse schema."""
		risk = state.risk_analysis or {}
		decision = state.final_decision or {}

		# Policy Retrieval format
		policy_matches_list = [
			{
				"rule_name": p["rule_name"],
				"rule_text": p["rule_text"],
				"status": p["status"],
				"source": p.get("source"),
				"score": p.get("score"),
			}
			for p in state.policy_matches
		]

		risk_val = state.final_risk_level or state.ml_risk_level or "Medium"
		risk_str = risk_val.value if hasattr(risk_val, "value") else risk_val

		report = {
			"borrower_name": state.borrower_input.full_name if state.borrower_input else "Unknown",
			"request_id": state.request_id,
			"risk_analysis": {
				"risk_level": risk_str,
				"ml_risk_score": state.ml_risk_score,
				"top_risk_factors": risk.get("top_risk_factors", []),
				"positive_factors": risk.get("positive_factors", []),
				"financial_sanity_score": risk.get("financial_sanity_score", 70.0),
				"policy_risk_score": state.policy_risk_score,
				"final_ai_score": state.final_ai_score,
				"arbitration_summary": state.arbitration_summary,
				"prediction_confidence": state.prediction_confidence,
				"disagreement_score": state.disagreement_score,
			},
			"policy_retrieval": {
				"rules_checked": len(state.policy_matches),
				"policies_matched": policy_matches_list,
			},
			"lending_decision": {
				"recommendation": decision.get("recommendation", "Manual Review"),
				"primary_reason": decision.get("primary_reason", "Policy analysis complete"),
				"secondary_reasons": decision.get("secondary_reasons", []),
				"suggested_action": decision.get("suggested_action", "Proceed with manual review"),
				"manual_review_needed": decision.get("manual_review_needed", True),
			},
			"foir": state.foir,
			"dti": state.dti,
			"proposed_emi": state.proposed_emi,
			"agent_interactions": state.agent_interactions,
			"score_breakdown": state.score_breakdown,
			"workflow_trace": state.workflow_trace,
		}

		return report

	def save_report(self, report: Dict[str, Any]) -> None:
		"""Cache report by request_id."""
		request_id = report.get("request_id")
		if request_id:
			self._cache[request_id] = report
			logger.info(f"💾 Report saved to cache for ID: {request_id}")

	def get_report(self, request_id: str) -> Dict[str, Any] | None:
		"""Retrieve cached report by request id."""
		report = self._cache.get(request_id)
		if report:
			logger.info(f"🔍 Report retrieved from cache for ID: {request_id}")
		else:
			logger.warning(f"⚠️ Report NOT found in cache for ID: {request_id}")
		return report


report_service = ReportService()
