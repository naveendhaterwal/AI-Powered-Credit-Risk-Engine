"""
Machine Learning Service
Loads and uses the credit risk prediction model.
"""

import joblib
import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Any

from app.core.config import settings
from app.schemas.borrower import BorrowerProfile

logger = logging.getLogger(__name__)


class MLService:
    """
    Service for ML model predictions.
    Loads multiple models and provides an ensemble consensus score.
    """
    
    def __init__(self):
        """Initialize the service - load models on startup"""
        self.models: Dict[str, Any] = {}
        self.feature_columns: List[str] = []
        self.preprocessor = None
        self.models_loaded = False
        self.use_fallback = False
        self._load_models()
    
    def _load_models(self):
        """Load multiple ML pipelines from disk"""
        try:
            model_dir = Path(settings.ML_MODEL_PATH).parent
            metadata_path = model_dir / "feature_metadata.json"
            
            if not metadata_path.exists():
                logger.warning("Feature metadata not found. Switching to rule-based fallback.")
                self.use_fallback = True
                return
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.feature_columns = metadata["expected_features"]
            
            # Map of model name to its expected filename
            model_registry = {
                "logistic": "logistic_pipeline.pkl",
                "random_forest": "rf_pipeline.pkl",
                "gradient_boosting": "gb_pipeline.pkl"
            }
            
            loaded_count = 0
            for name, filename in model_registry.items():
                path = model_dir / filename
                if path.exists():
                    try:
                        self.models[name] = joblib.load(path)
                        logger.info(f"Loaded {name} pipeline from {path}")
                        loaded_count += 1
                    except Exception as e:
                        logger.error(f"Error loading {name} pipeline: {e}")
            
            if loaded_count == 0:
                logger.warning("No ML pipelines found. Switching to rule-based fallback.")
                self.use_fallback = True
            else:
                self.models_loaded = True
                logger.info(f"ML Service initialized with {loaded_count} calibrated pipelines.")
                
        except Exception as e:
            logger.error(f"Error in ML Service initialization: {str(e)}. Enabling rule-based fallback.")
            self.models = {}
            self.feature_columns = []
            self.models_loaded = False
            self.use_fallback = True

    def _loan_purpose_to_model(self, purpose: str) -> str:
        """Map app loan purpose to dataset categories."""
        purpose_map = {
            "Home": "p1",
            "Auto": "p2",
            "Personal": "p3",
            "Business": "p4",
        }
        return purpose_map.get(purpose, "p3")

    def _build_model_input(self, borrower: BorrowerProfile) -> pd.DataFrame:
        """Build a one-row DataFrame mapped to raw training features before preprocessing."""
        if not self.feature_columns:
            raise ValueError("Feature columns are not loaded")

        loan_amount = float(borrower.loan_amount_requested)
        income = float(borrower.monthly_income)
        temp_income = income if income > 0 else np.nan
        
        # Calculate engineered features matching training script
        loan_to_income = loan_amount / (temp_income * 12 + 1) if temp_income is not np.nan else np.nan
        term = int(borrower.loan_tenure_months)
        
        # Assume 5% flat rate for EMI calculation if missing
        annual_rate = 0.05
        monthly_rate = annual_rate / 12
        estimated_emi = loan_amount * (monthly_rate * (1 + monthly_rate)**term) / ((1 + monthly_rate)**term - 1)
        
        derived_foir = estimated_emi / (temp_income + 1) if temp_income is not np.nan else np.nan
        property_value = float(loan_amount / 0.8) if loan_amount > 0 else 0.0
        loan_to_property_ratio = loan_amount / (property_value + 1)
        
        # Clip features
        ltv_raw = loan_amount / property_value * 100 if property_value > 0 else 80.0
        ltv_clipped = float(min(150.0, max(0.0, ltv_raw)))
        dti_clipped = float(min(65.0, max(0.0, borrower.dti * 100.0)))
        
        row = {
            "loan_amount": loan_amount,
            "term": term,
            "property_value": property_value,
            "income": temp_income,
            "Credit_Score": int(borrower.credit_score),
            "LTV": ltv_clipped,
            "dtir1": dti_clipped,
            "loan_to_income": loan_to_income,
            "estimated_emi": estimated_emi,
            "derived_foir": derived_foir,
            "loan_to_property_ratio": loan_to_property_ratio,
            
            "loan_purpose": self._loan_purpose_to_model(borrower.loan_purpose),
            "Credit_Worthiness": "l1" if borrower.credit_score >= 700 else "l2",
            "business_or_commercial": "b/c" if borrower.loan_purpose == "Business" else "nob/c",
            "loan_limit": "cf",
            "approv_in_adv": "nopre",
            "loan_type": "type1",
            "Neg_ammortization": "not_neg",
            "interest_only": "not_int",
            "lump_sum_payment": "not_lpsm",
            "occupancy_type": "pr",
            "total_units": "1U",
            "credit_type": "EXP",
            "co-applicant_credit_type": "CIB",
            "submission_of_application": "to_inst",
            "Region": "south"
        }

        # Ensure ordered as training features
        ordered_row = {col: row.get(col, np.nan) for col in self.feature_columns}
        return pd.DataFrame([ordered_row])
        
    def _prepare_features(self, borrower: BorrowerProfile) -> pd.DataFrame:
        """Build the raw DataFrame for the ML pipeline."""
        return self._build_model_input(borrower)
            
    def _predict_fallback(self, borrower: BorrowerProfile) -> Tuple[str, float, float, Dict[str, Any]]:
        """Rule-based risk calculation as a fallback for missing or incompatible ML model."""
        logger.info("Using rule-based risk prediction fallback")
        base_risk = 100 - ((borrower.credit_score - 300) / 600 * 100)
        foir_penalty = max(0, (borrower.foir - 0.4) * 100)
        dti_penalty = max(0, (borrower.dti - 0.4) * 50)
        employment_bonus = -5 if borrower.employment_type == "Salaried" else 5
        
        total_risk = base_risk + foir_penalty + dti_penalty + employment_bonus
        total_risk = max(1.0, min(100, total_risk))
        
        risk_level, _ = self._interpret_prediction(total_risk)
        
        score_breakdown = {
            "method": "rule_based_fallback",
            "components": {
                "base_risk_from_credit_score": round(base_risk, 4),
                "foir_penalty": round(foir_penalty, 4),
                "dti_penalty": round(dti_penalty, 4),
            },
        }


    
    def detect_ood(self, borrower: BorrowerProfile) -> Tuple[bool, List[str]]:
        """
        Detect Out-of-Distribution (OOD) scenarios using expert financial boundaries.
        """
        reasons = []
        # Extreme Financial Ratios
        if borrower.foir > 1.2: reasons.append("Extreme FOIR (>120%) detected")
        if borrower.dti > 0.8: reasons.append("Extreme DTI (>80%) detected")
        
        # Unrealistic Borrowers
        if borrower.credit_score < 300 or borrower.credit_score > 900:
            reasons.append(f"Invalid Credit Score ({borrower.credit_score})")
        
        # Extreme Loan Size Relative to Income
        annual_income = borrower.monthly_income * 12
        if annual_income > 0 and (borrower.loan_amount_requested / annual_income) > 20:
            reasons.append("Loan amount exceeds 20x annual income")
            
        return len(reasons) > 0, reasons

    def calculate_confidence(self, 
        models_active: int, 
        disagreement_score: float, 
        is_ood: bool, 
        fallback_level: int
    ) -> Tuple[float, List[str]]:
        """
        Governance logic for confidence calculation.
        """
        base = 100.0
        reasons = []
        
        # Penalty for inactive models
        if models_active < 3:
            penalty = (3 - models_active) * 20
            base -= penalty
            reasons.append(f"Reduced by {penalty}% due to degraded ensemble ({models_active}/3 active)")
            
        # Penalty for model disagreement
        if disagreement_score > 30:
            penalty = min(25, disagreement_score / 2)
            base -= penalty
            reasons.append(f"Reduced by {round(penalty, 1)}% due to high model disagreement")
            
        # Penalty for OOD
        if is_ood:
            base -= 40
            reasons.append("Critical reduction (40%) due to Out-of-Distribution data")
            
        # Penalty for fallback usage
        if fallback_level >= 3:
            base -= 15
            reasons.append("Reduced by 15% due to fallback logic usage")
            
        return max(10.0, base), reasons

    def _interpret_prediction(self, prediction_100: float) -> Tuple[str, float]:
        """Interpret risk score on 0-100 scale."""
        if prediction_100 < 40:
            risk_level = "Low"
        elif prediction_100 < 65:
            risk_level = "Medium"
        else:
            risk_level = "High"
        return risk_level, prediction_100

    def predict_all_models(self, borrower: BorrowerProfile) -> Dict[str, Any]:
        """
        Predict using all models and return ensemble details.
        Implements hard banking rules, ensemble averaging, and intelligent fallback.
        """
        # ==========================================================
        # LEVEL 3: HARD BANKING RULES (Deterministic Pre-ML Override)
        # ==========================================================
        calculated_ltv = (borrower.loan_amount_requested / (borrower.loan_amount_requested / 0.8)) * 100 if borrower.loan_amount_requested > 0 else 0
        
        if borrower.credit_score < 550:
            return self._build_deterministic_override("Reject", ["Low Credit Score"], "Credit score below 550 threshold for standard underwriting.")
        if calculated_ltv > 120:
            return self._build_deterministic_override("Reject", ["High LTV"], "Loan-to-Value ratio exceeds 120% regulatory limit.")
        if borrower.foir * 100 > 60:
            return self._build_deterministic_override("Reject", ["Extreme FOIR"], "FOIR exceeds 60% internal safety threshold.")
        if borrower.monthly_income <= 0:
            return self._build_deterministic_override("Manual Review", ["Zero Income"], "Income reporting error: requires manual verification.")

        # ==========================================================
        # LEVELS 1-2: ENSEMBLE ML EXECUTION
        # ==========================================================
        base_weights = {
            "logistic": 0.25,
            "random_forest": 0.40,
            "gradient_boosting": 0.35
        }
        
        models_output = {
            "logistic": {"status": "failed", "reason": "model_loading_failure"},
            "random_forest": {"status": "failed", "reason": "model_loading_failure"},
            "gradient_boost": {"status": "failed", "reason": "model_loading_failure"}
        }

        name_map = {
            "logistic": "logistic",
            "random_forest": "random_forest",
            "gradient_boosting": "gradient_boost"
        }
        
        if self.use_fallback or not self.models_loaded:
            return self._execute_level_4_fallback(borrower, models_output, "model_loading_failure")

        try:
            features = self._prepare_features(borrower)
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return self._execute_level_4_fallback(borrower, models_output, "preprocessing_error")

        successful_models = {}
        
        for name, model in self.models.items():
            if name not in base_weights:
                continue
                
            out_name = name_map[name]
            try:
                probs = model.predict_proba(features)[0]
                pos_idx = 1
                if hasattr(model, "classes_"):
                    classes = list(model.classes_)
                    if 1 in classes:
                        pos_idx = classes.index(1)
                    elif '1' in classes:
                        pos_idx = classes.index('1')
                
                score = float(probs[pos_idx]) * 100
                successful_models[name] = score
                models_output[out_name] = {"status": "healthy", "score": round(score, 2)}
            except Exception as e:
                models_output[out_name] = {"status": "failed", "reason": str(e)[:30]}
                logger.error(f"Error predicting with {name}: {e}")

        models_active = len(successful_models)
        
        # Level 4 Fallback if entirely failed
        if models_active == 0:
            return self._execute_level_4_fallback(borrower, models_output, "all_models_failed_inference")
            
        total_remaining_weight = sum(base_weights[name] for name in successful_models)
        final_score = 0.0
        for name, score in successful_models.items():
            dynamic_weight = base_weights[name] / total_remaining_weight
            final_score += score * dynamic_weight
            
        if models_active == 3:
            fallback_level = 1
            ensemble_health = "healthy"
            base_confidence = 0.95
        elif models_active == 2:
            fallback_level = 2
            ensemble_health = "degraded"
            base_confidence = 0.65
        else:
            fallback_level = 3
            ensemble_health = "critical"
            base_confidence = 0.35
            
        category, _ = self._interpret_prediction(final_score)

        # Confidence & OOD Governance
        is_ood, ood_reasons = self.detect_ood(borrower)
        
        disagreement_score = 0.0
        if models_active > 1:
            scores_list = list(successful_models.values())
            disagreement_score = max(scores_list) - min(scores_list)
            
        confidence_val, confidence_reasons = self.calculate_confidence(
            models_active, disagreement_score, is_ood, fallback_level
        )

        return {
            "ml_ensemble_score": round(final_score, 2), # 0-100 scale
            "override_triggered": False,
            "critical_flags": ood_reasons,
            "ood_flag": is_ood,
            "risk_level": category,
            "prediction_confidence": confidence_val,
            "confidence_reasoning": confidence_reasons,
            "disagreement_score": round(disagreement_score, 2),
            "fallback_level_used": fallback_level,
            "ensemble_health": ensemble_health,
            "models": models_output,
            "recommendation": "Pending AI Arbitration",
            "reasoning": "Ensemble statistical baseline established."
        }

    def _execute_level_4_fallback(self, borrower: BorrowerProfile, models_output: dict, reason: str) -> Dict[str, Any]:
        """Execute Level 4 deterministic fallback when ML ensemble completely fails."""
        risk_level, total_risk, _, _ = self._predict_fallback(borrower)
        if not risk_level.endswith("Risk"):
            risk_level += " Risk"
            
        return {
            "ml_ensemble_score": round(total_risk, 2),
            "override_triggered": False,
            "critical_flags": ["Model Inference Failed", reason],
            "ood_flag": False,
            "risk_level": risk_level,
            "prediction_confidence": 20.0,
            "confidence_reasoning": ["Forced reduction due to system failure"],
            "disagreement_score": 0.0,
            "fallback_level_used": 4,
            "ensemble_health": "failed",
            "models": models_output,
            "recommendation": "Manual Review",
            "reasoning": f"ML pipeline unavailable ({reason}). Rule-based assessment used."
        }
        
    def _build_deterministic_override(self, recommendation: str, flags: List[str], reasoning: str) -> Dict[str, Any]:
        """Bypass ML to enforce deterministic banking limits."""
        return {
            "ml_ensemble_score": 100.0 if recommendation == "Reject" else 60.0,
            "override_triggered": True,
            "critical_flags": flags,
            "ood_flag": True,
            "risk_level": "High Risk",
            "prediction_confidence": 100.0, 
            "confidence_reasoning": ["Deterministic banking threshold violation"],
            "disagreement_score": 0.0,
            "fallback_level_used": 3,
            "ensemble_health": "healthy",
            "models": {},
            "recommendation": recommendation,
            "reasoning": reasoning
        }

# Global singleton
ml_service = MLService()
