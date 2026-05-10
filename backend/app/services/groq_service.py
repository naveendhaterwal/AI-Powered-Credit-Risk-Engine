"""
Groq LLM Service
Handles calls to Groq API for AI agent responses.
"""

import logging
import json
from typing import Optional

from groq import Groq
from app.core.config import settings

logger = logging.getLogger(__name__)


class GroqService:
    """
    Service for interacting with Groq LLM.
    Used by AI agents to generate analysis and decisions.
    """
    
    def __init__(self):
        """Initialize Groq client"""
        self.client = None
        self.model = settings.GROQ_MODEL
        self._initialize_client()
    
    def _initialize_client(self):
        """Create Groq client"""
        try:
            if not settings.GROQ_API_KEY:
                logger.warning("GROQ_API_KEY not set in environment")
                return
            
            self.client = Groq(api_key=settings.GROQ_API_KEY)
            logger.info("✓ Groq client initialized")
            
        except Exception as e:
            logger.error(f"Error initializing Groq client: {str(e)}")
            self.client = None
    
    def call_llm(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 500
    ) -> str:
        """
        Call Groq LLM with a prompt.
        
        Args:
            prompt: User message
            system_prompt: System context
            temperature: Creativity (0=deterministic, 1=creative)
            max_tokens: Max response length
            
        Returns:
            Response text from LLM
        """
        
        try:
            # Strict mode: caller decides fallback behavior.
            if not self.client:
                raise RuntimeError("Groq client not initialized")
            
            # Build messages
            messages = []
            
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            messages.append({"role": "user", "content": prompt})
            
            # Call Groq API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.95,
                stream=False
            )
            
            # Extract response text
            result = response.choices[0].message.content
            
            logger.info(f"LLM Response: {len(result)} characters")
            return result
            
        except Exception as e:
            logger.error(f"Error calling Groq API: {str(e)}")
            raise

    def _extract_json(self, text: str) -> dict:
        """Helper to extract and parse JSON from LLM response strings which may include markdown tags."""
        logger.debug(f"Attempting to extract JSON from: {text[:100]}...")
        clean_text = text.strip()
        
        # Remove markdown code blocks if present
        if clean_text.startswith("```"):
            # Look for the start of the actual JSON content
            first_brace = clean_text.find("{")
            last_brace = clean_text.rfind("}")
            if first_brace != -1 and last_brace != -1:
                clean_text = clean_text[first_brace : last_brace + 1]
        
        try:
            return json.loads(clean_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from cleaned text: {e}")
            # Fallback attempt: find the first { and last } if not already tried
            first_brace = clean_text.find("{")
            last_brace = clean_text.rfind("}")
            if first_brace != -1 and last_brace != -1:
                try:
                    return json.loads(clean_text[first_brace : last_brace + 1])
                except:
                    pass
            raise


class RiskAnalysisAgent:
    """Agent for analyzing credit risk"""
    
    def __init__(self, groq_service: GroqService):
        self.groq = groq_service
    
    def analyze(self, borrower_data: dict, risk_score: float, risk_level: str) -> dict:
        """
        Analyze risk factors for borrower.
        
        Returns dict with top_risk_factors, positive_factors, etc.
        """
        
        system_prompt = """You are a credit risk analyst.
Analyze the borrower's profile and provide specific risk factors.
Format your response as a flat JSON object with these EXACT keys:
- "top_risk_factors": (List of strings)
- "positive_factors": (List of strings)
- "confidence_score": (Float between 0 and 1)
- "risk_level": (String: "Low", "Medium", or "High")

Do not nest these keys inside another object."""
        
        user_prompt = f"""
Analyze this borrower's credit risk:
- Age: {borrower_data.get('age')}
- Credit Score: {borrower_data.get('credit_score')}
- Monthly Income: {borrower_data.get('monthly_income')}
- Employment: {borrower_data.get('employment_type')}
- FOIR: {borrower_data.get('foir')}
- DTI: {borrower_data.get('dti')}
- Loan Amount: {borrower_data.get('loan_amount_requested')}

ML Risk Score: {risk_score}
ML Risk Level: {risk_level}

Provide the analysis as a JSON object."""
        
        try:
            response = self.groq.call_llm(user_prompt, system_prompt, temperature=0.1)
            raw_result = self.groq._extract_json(response)
            
            # --- NORMALIZATION LAYER ---
            # Handle potential nesting (e.g., creditRiskAnalysis)
            if "creditRiskAnalysis" in raw_result:
                raw_result.update(raw_result.pop("creditRiskAnalysis"))
            
            # Map camelCase to snake_case
            mapping = {
                "topRiskFactors": "top_risk_factors",
                "topPositiveFactors": "positive_factors",
                "positiveFactors": "positive_factors",
                "confidenceScore": "confidence_score",
                "riskLevel": "risk_level"
            }
            
            result = {}
            for k, v in raw_result.items():
                target_key = mapping.get(k, k)
                result[target_key] = v
            
            # Ensure required keys exist
            result.setdefault("top_risk_factors", [])
            result.setdefault("positive_factors", [])
            result.setdefault("confidence_score", 0.7)
            
            # Flatten objects to strings if needed (LLM sometimes returns objects in factors)
            for factor_key in ["top_risk_factors", "positive_factors"]:
                processed = []
                for item in result.get(factor_key, []):
                    if isinstance(item, dict):
                        # Extract "factor" or "description" string
                        processed.append(item.get("factor") or item.get("description") or str(item))
                    else:
                        processed.append(str(item))
                result[factor_key] = processed

            result["agent_source"] = "groq"
            result["warnings"] = []
            
            # Metadata for logging
            interaction = {
                "agent": "Risk Analysis Agent",
                "system_prompt": system_prompt,
                "prompt": user_prompt,
                "response": response
            }
            return result, interaction
        except Exception as exc:
            if settings.STRICT_NO_FALLBACKS:
                raise RuntimeError(f"Risk Analysis Agent failed in strict mode: {exc}") from exc

            # Deterministic fallback without fabricated narrative text.
            risk_factors = []
            positive_factors = []
            
            # ... (omitted for brevity in chunking, will keep existing fallback logic)
            # Actually I need to include it or it will be deleted.
            if borrower_data.get("foir", 0.0) > 0.45:
                risk_factors.append("FOIR above 45% threshold")
            else:
                positive_factors.append("FOIR within threshold")

            if borrower_data.get("dti", 0.0) > 0.40:
                risk_factors.append("DTI above 40% threshold")
            else:
                positive_factors.append("DTI within threshold")

            credit_score = int(borrower_data.get("credit_score", 0))
            if credit_score < 600:
                risk_factors.append("Credit score below 600")
            elif credit_score >= 700:
                positive_factors.append("Credit score is strong")

            if not risk_factors:
                risk_factors.append("No major threshold breach detected")

            if not positive_factors:
                positive_factors.append("Limited positive factors from current threshold checks")

            fallback_result = {
                "top_risk_factors": risk_factors,
                "positive_factors": positive_factors,
                "confidence_score": 0.70,
                "agent_source": "fallback",
                "warnings": [f"Risk analysis fallback used: {exc}"],
            }
            interaction = {
                "agent": "Risk Analysis Agent",
                "system_prompt": system_prompt,
                "prompt": user_prompt,
                "response": json.dumps(fallback_result, indent=2),
                "error": str(exc)
            }
            return fallback_result, interaction


class LendingDecisionAgent:
    """Agent for making lending decisions"""
    
    def __init__(self, groq_service: GroqService):
        self.groq = groq_service
    
    def decide(
        self,
        model_scores: dict,
        ensemble_score: float,
        disagreement_flag: str,
        policy_matches: list,
        borrower_data: dict
    ) -> dict:
        """
        Make final lending decision based on risk and policies.
        
        Returns dict with recommendation, reasoning.
        """
        
        system_prompt = """You are a lending decision expert.
Make clear, justifiable lending decisions based on the ensemble model scores, risk profile, and policy matches.
If the models disagree (High Disagreement), you MUST explain the differences between the models.
You MUST justify your final recommendation using both the policy rules and the ML scores.
Format your response as a flat JSON object with these EXACT keys:
- "recommendation": (String: "Approve", "Approve with conditions", "Manual Review", or "Reject")
- "reasoning": (String, detailed explanation of the differences between models (if any) and justification for the final recommendation)

Do not nest these keys inside another object."""
        
        user_prompt = f"""
Make a lending decision for:
Ensemble Risk Score: {ensemble_score}
Model Disagreement Flag: {disagreement_flag}
Individual Model Scores: {json.dumps(model_scores, indent=2)}

Borrower Profile:
- FOIR: {borrower_data.get('foir')}
- DTI: {borrower_data.get('dti')}
- Credit Score: {borrower_data.get('credit_score')}
- Monthly Income: {borrower_data.get('monthly_income')}
- Requested Loan: {borrower_data.get('loan_amount_requested')}

Policy Violations / Compliances:
{json.dumps([p.get('rule_text') for p in policy_matches], indent=2)}

Provide the final decision as a JSON object."""
        
        try:
            response = self.groq.call_llm(user_prompt, system_prompt, temperature=0.1)
            raw_result = self.groq._extract_json(response)
            
            # --- NORMALIZATION LAYER ---
            result = {}
            result["recommendation"] = raw_result.get("recommendation", "Manual Review")
            result["reasoning"] = raw_result.get("reasoning", "Decision generated by AI agent")

            result.setdefault("recommendation", "Manual Review")
            result.setdefault("reasoning", "Decision generated by AI agent")

            result["agent_source"] = "groq"
            result["warnings"] = []
            
            interaction = {
                "agent": "Decision Agent",
                "system_prompt": system_prompt,
                "prompt": user_prompt,
                "response": response
            }
            return result, interaction
        except Exception as exc:
            if settings.STRICT_NO_FALLBACKS:
                raise RuntimeError(f"Lending Decision Agent failed in strict mode: {exc}") from exc

            violations = len([p for p in policy_matches if p.get("status") == "Violated"])
            if violations >= 2:
                recommendation = "Reject"
                reason = "Multiple policy violations detected"
            elif violations == 1:
                recommendation = "Manual Review"
                reason = "Single policy violation requires underwriter review"
            else:
                recommendation = "Approve"
                reason = "No policy violations detected"

            fallback_result = {
                "recommendation": recommendation,
                "reasoning": reason,
                "agent_source": "fallback",
                "warnings": [f"Decision fallback used: {exc}"],
            }
            interaction = {
                "agent": "Decision Agent",
                "system_prompt": system_prompt,
                "prompt": user_prompt,
                "response": json.dumps(fallback_result, indent=2),
                "error": str(exc)
            }
            return fallback_result, interaction


class ChatAgent:
    """Agent for interactive chat about a credit report"""
    
    def __init__(self, groq_service: GroqService):
        self.groq = groq_service
        
    def chat(self, report_data: dict, user_message: str) -> dict:
        # Filter report data to stay under token limits
        essential_data = {
            "borrower_name": report_data.get("borrower_name"),
            "risk_analysis": report_data.get("risk_analysis"),
            "policy_matches": report_data.get("policy_retrieval", {}).get("policies_matched", []),
            "lending_decision": report_data.get("lending_decision"),
            "financials": {
                "foir": report_data.get("foir"),
                "dti": report_data.get("dti"),
                "proposed_emi": report_data.get("proposed_emi")
            }
        }
        
        system_prompt = f"""You are a helpful credit risk advisor. 
Discuss this loan decision based ONLY on this data:
{json.dumps(essential_data, indent=2)}

Answer concisely and professionally."""

        try:
            response = self.groq.call_llm(user_message, system_prompt, temperature=0.3, max_tokens=1000)
            return {"answer": response, "model_source": "groq"}
        except Exception as exc:
            # Check if it's a rate limit / token limit error to provide a better message
            error_msg = str(exc)
            if "Limit 6000" in error_msg or "tokens" in error_msg:
                return {"answer": "The report is too large for the current AI model limit. Please try a shorter question or contact support.", "model_source": "error"}
            return {"answer": f"I'm sorry, I cannot analyze the report right now.", "model_source": "fallback"}

class AIScoringAgent:
    """Agent for synthesizing final risk score based on ML, policies, and strengths"""
    
    def __init__(self, groq_service: GroqService):
        self.groq = groq_service
        
    def evaluate(
        self, 
        ml_score: float, 
        ml_level: str,
        risk_analysis: dict,
        policy_matches: list,
        borrower_data: dict,
        model_scores: Optional[dict] = None
    ) -> dict:
        """
        Produce a final AI-synthesized risk score.
        
        Returns dict with final_score (0-100) and reasoning.
        """
        
        system_prompt = """You are a senior credit risk evaluator.
Your task is to provide the FINAL risk score for a loan applicant.
You must take the baseline Machine Learning (ML) score and refine it using:
1. Company lending policies (retrieved as chunks).
2. Qualitative strengths and weaknesses.
3. Financial ratios (FOIR, DTI).

The ML score is a mathematical baseline, but your AI score is the final human-like judgment that considers policy nuances.

CRITICAL: If there are policy violations (e.g., FOIR > 50% for salaried), you MUST increase the risk score significantly (at least 60-70 or higher), even if the ML baseline is low. Never return 0.0 if there are active weaknesses or policy failures.

Format your response as a flat JSON object with these EXACT keys:
- "final_score": (Float between 0 and 100, where 0 is safest and 100 is riskiest)
- "reasoning": (String, detailed explanation of how the ML score was adjusted based on company policy and borrower profile)

Do not nest these keys."""

        user_prompt = f"""
ML Consensus Breakdown:
{json.dumps(model_scores or {"baseline_model": ml_score}, indent=2)}

Ensemble Baseline Score: {ml_score} ({ml_level})
Positive strengths: {json.dumps(risk_analysis.get('positive_factors', []))}
Negative weaknesses: {json.dumps(risk_analysis.get('top_risk_factors', []))}

Company Policy Snippets:
{json.dumps([p.get('rule_text') for p in policy_matches], indent=2)}

Borrower Ratios:
- FOIR: {borrower_data.get('foir')}
- DTI: {borrower_data.get('dti')}

Calculated final risk score based on the above context. If models disagree (e.g., Logistic Regression says Low but Random Forest says High), explain your reasoning for the chosen final score in the 'reasoning' field.
"""

        try:
            response = self.groq.call_llm(user_prompt, system_prompt, temperature=0.1)
            result = self.groq._extract_json(response)
            
            # Normalization
            final_score = result.get("final_score", ml_score)
            reasoning = result.get("reasoning", "Score determined by AI synthesis of ML and policy context.")
            
            interaction = {
                "agent": "AI Scoring Agent",
                "system_prompt": system_prompt,
                "prompt": user_prompt,
                "response": response
            }
            return {"final_score": final_score, "reasoning": reasoning}, interaction
            
        except Exception as exc:
            logger.error(f"Error in AI Scoring Agent: {exc}")
            # Fallback to ML score
            fallback_result = {
                "final_score": ml_score,
                "reasoning": f"AI Scoring failed, falling back to ML baseline. Error: {str(exc)}"
            }
            interaction = {
                "agent": "AI Scoring Agent",
                "error": str(exc),
                "response": "Fallback to ML score"
            }
            return fallback_result, interaction


# Global instances
groq_service = GroqService()
risk_agent = RiskAnalysisAgent(groq_service)
scoring_agent = AIScoringAgent(groq_service)
decision_agent = LendingDecisionAgent(groq_service)
chat_agent = ChatAgent(groq_service)
