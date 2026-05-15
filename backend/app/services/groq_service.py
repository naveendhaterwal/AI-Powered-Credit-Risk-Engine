"""
Groq LLM Service
Orchestrates multiple AI agents for risk analysis, policy evaluation, and lending decisions.
"""

import os
import json
import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from groq import Groq
from app.core.config import settings

logger = logging.getLogger(__name__)

class GroqService:
    """Base service for interacting with Groq Cloud API."""
    
    def __init__(self):
        self.api_key = settings.GROQ_API_KEY
        if not self.api_key:
            logger.warning("⚠️ GROQ_API_KEY not found in settings")
        self.client = Groq(api_key=self.api_key)
        self.model = settings.GROQ_MODEL

    def call_llm(self, prompt: str, system_prompt: str = "You are a helpful assistant.", temperature: float = 0.2) -> str:
        """Call Groq API with given prompts."""
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                temperature=temperature,
                response_format={"type": "json_object"}
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            logger.error(f"❌ Groq API Call Failed: {str(e)}")
            raise e

    def _extract_json(self, text: str) -> dict:
        """Parse JSON from LLM response."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Fallback for malformed JSON
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except:
                    pass
            return {"error": "Failed to parse JSON"}

class RiskAnalysisAgent:
    """Agent for deep financial borrower analysis."""
    
    def __init__(self, groq_service: GroqService):
        self.groq = groq_service
    
    def analyze(self, borrower_data: dict, ml_score: float) -> Tuple[dict, dict]:
        system_prompt = """You are a Senior Credit Risk Analyst. 
Your role is to analyze the borrower's financial profile with clinical precision.
Focus on:
1. Debt-to-Income (DTI) and Fixed Obligation (FOIR) ratios.
2. Income stability vs Loan burden.
3. Creditworthiness metrics.

Output must be concise, professional, and audit-ready. Avoid generic filler.
Return a JSON object with:
- "top_risk_factors": List[str]
- "positive_factors": List[str]
- "analysis_summary": str
- "financial_sanity_score": float (0-100, where 100 is perfectly healthy)"""

        user_prompt = f"""Analyze the following borrower financials:
{json.dumps(borrower_data, indent=2)}
Baseline ML Risk Score: {ml_score}"""
        
        try:
            response = self.groq.call_llm(user_prompt, system_prompt, temperature=0.1)
            result = self.groq._extract_json(response)
            interaction = {"agent": "Risk Analysis Agent", "response": response}
            return result, interaction
        except Exception as exc:
            return {"top_risk_factors": ["Analysis failure"], "financial_sanity_score": 50.0}, {"error": str(exc)}

class PolicyAgent:
    """Agent for compliance and policy evaluation."""
    
    def __init__(self, groq_service: GroqService):
        self.groq = groq_service
        
    def evaluate(self, policy_matches: list, borrower_data: dict) -> Tuple[dict, dict]:
        system_prompt = """You are a Banking Compliance Officer.
Your role is to evaluate the borrower against internal lending policies.
Focus on:
1. Critical violations.
2. Compliance exceptions.
3. Policy-based risk weighting.

Return a JSON object with:
- "policy_risk_score": float (0-100, 100 is maximum violation)
- "critical_violations": List[str]
- "compliance_summary": str
- "severity_level": str ("Critical", "Warning", "Info")"""

        user_prompt = f"""Evaluate policy compliance for:
{json.dumps(borrower_data, indent=2)}
Retrieved Policy Snippets:
{json.dumps(policy_matches, indent=2)}"""

        try:
            response = self.groq.call_llm(user_prompt, system_prompt, temperature=0.1)
            result = self.groq._extract_json(response)
            interaction = {"agent": "Policy Agent", "response": response}
            return result, interaction
        except Exception as exc:
            return {"policy_risk_score": 0.0, "critical_violations": []}, {"error": str(exc)}

class ArbitrationAgent:
    """Agent for resolving conflicts between ML and Policies."""
    
    def __init__(self, groq_service: GroqService):
        self.groq = groq_service
        
    def arbitrate(self, ml_data: dict, risk_analysis: dict, policy_data: dict) -> Tuple[dict, dict]:
        system_prompt = """You are a Principal Risk Arbitrator on the Credit Committee.
Your role is to weigh statistical ML predictions against policy compliance and system reliability.
You must decide if the ML output is trustworthy or if policy violations necessitate an override.

Analyze Model Disagreement and Confidence Degradation.
Return a JSON object with:
- "final_ai_score": float (0-100)
- "arbitration_summary": str (Justify why ML score was adjusted or upheld)
- "trust_level": str ("High", "Degraded", "Low")
- "escalation_required": bool"""

        user_prompt = f"""Perform risk arbitration:
ML Ensemble Data: {json.dumps(ml_data, indent=2)}
Risk Analysis: {json.dumps(risk_analysis, indent=2)}
Policy Evaluation: {json.dumps(policy_data, indent=2)}"""

        try:
            response = self.groq.call_llm(user_prompt, system_prompt, temperature=0.1)
            result = self.groq._extract_json(response)
            interaction = {"agent": "Arbitration Agent", "response": response}
            return result, interaction
        except Exception as exc:
            return {"final_ai_score": ml_data.get('ml_ensemble_score', 50.0)}, {"error": str(exc)}

class LendingDecisionAgent:
    """Agent for final operational recommendation."""
    
    def __init__(self, groq_service: GroqService):
        self.groq = groq_service
        
    def decide(self, arbitration_data: dict, risk_level: str) -> Tuple[dict, dict]:
        system_prompt = """You are the Chief Lending Officer.
Your role is to issue the final operational recommendation.
Your reasoning must be concise, ratio-driven, and auditor-ready.

Return a JSON object with:
- "recommendation": "Approve" | "Approve with Conditions" | "Manual Review" | "Reject"
- "primary_reason": str (Executive reasoning)
- "secondary_reasons": List[str]
- "suggested_action": str"""

        user_prompt = f"""Finalize lending decision:
Arbitration Summary: {arbitration_data.get('arbitration_summary')}
Risk Level: {risk_level}
Arbitrated Score: {arbitration_data.get('final_ai_score')}"""

        try:
            response = self.groq.call_llm(user_prompt, system_prompt, temperature=0.1)
            result = self.groq._extract_json(response)
            interaction = {"agent": "Decision Agent", "response": response}
            return result, interaction
        except Exception as exc:
            return {"recommendation": "Manual Review"}, {"error": str(exc)}

class ChatAgent:
    """Agent for interactive Q&A about the report."""
    def __init__(self, groq_service: GroqService):
        self.groq = groq_service
        
    def chat(self, message: str, report_data: dict) -> dict:
        system_prompt = "You are a Credit Risk Advisor. Answer questions about the credit report."
        user_prompt = f"Report Data: {json.dumps(report_data)}\nQuestion: {message}"
        try:
            response = self.groq.call_llm(user_prompt, system_prompt)
            return {"answer": response, "model_source": "groq"}
        except:
            return {"answer": "I'm sorry, I couldn't process that question.", "model_source": "fallback"}

# Global instances
groq_service = GroqService()
risk_agent = RiskAnalysisAgent(groq_service)
policy_agent = PolicyAgent(groq_service)
arbitration_agent = ArbitrationAgent(groq_service)
decision_agent = LendingDecisionAgent(groq_service)
chat_agent = ChatAgent(groq_service)
