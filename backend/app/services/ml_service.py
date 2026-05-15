"""
Machine Learning Service
Loads and uses the credit risk prediction model.
"""

import joblib
import logging
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
        """Load multiple ML models and preprocessing pipeline from disk"""
        try:
            feature_path = Path(settings.FEATURE_COLUMNS_PATH)
            preprocessor_path = Path(settings.ML_MODEL_PATH).parent / "preprocessing_pipeline.pkl"
            
            if not feature_path.exists() or not preprocessor_path.exists():
                logger.warning("Models, features, or preprocessor not found. Switching to rule-based fallback.")
                self.use_fallback = True
                return
            
            self.feature_columns = joblib.load(feature_path)
            self.preprocessor = joblib.load(preprocessor_path)
            
            # Map of model name to its expected filename
            model_registry = {
                "logistic": "logistic_model.pkl",
                "random_forest": "rf_model.pkl",
                "gradient_boosting": "gb_model.pkl"
            }
            
            loaded_count = 0
            for name, filename in model_registry.items():
                path = Path(settings.ML_MODEL_PATH).parent / filename
                if path.exists():
                    try:
                        self.models[name] = joblib.load(path)
                        logger.info(f"Loaded {name} model from {path}")
                        loaded_count += 1
                    except Exception as e:
                        logger.error(f"Error loading {name} model: {e}")
            
            if loaded_count == 0:
                logger.warning("No ML models found. Switching to rule-based fallback.")
                self.use_fallback = True
            else:
                self.models_loaded = True
                logger.info(f"ML Service initialized with {loaded_count} models and preprocessing pipeline.")
                
        except Exception as e:
            logger.error(f"Error in ML Service initialization: {str(e)}. Enabling rule-based fallback.")
            self.models = {}
            self.feature_columns = []
            self.preprocessor = None
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
        
    def _prepare_features(self, borrower: BorrowerProfile) -> np.ndarray:
        """Prepare borrower data using the scikit-learn preprocessing pipeline."""
        df = self._build_model_input(borrower)
        if self.preprocessor:
            return self.preprocessor.transform(df)
        return df.to_numpy()
            
    def _predict_fallback(self, borrower: BorrowerProfile) -> Tuple[str, float, float, Dict[str, Any]]:
        """Rule-based risk calculation as a fallback for missing or incompatible ML model."""
        logger.info("Using rule-based risk prediction fallback")
        base_risk = 100 - ((borrower.credit_score - 300) / 600 * 100)
        foir_penalty = max(0, (borrower.foir - 0.4) * 100)
        dti_penalty = max(0, (borrower.dti - 0.4) * 50)
        employment_bonus = -5 if borrower.employment_type == "Salaried" else 5
        
        total_risk = base_risk + foir_penalty + dti_penalty + employment_bonus
        total_risk = max(1.0, min(100, total_risk))
        
        risk_level, _ = self._interpret_prediction(total_risk / 100.0)
        
        score_breakdown = {
            "method": "rule_based_fallback",
            "components": {
                "base_risk_from_credit_score": round(base_risk, 4),
                "foir_penalty": round(foir_penalty, 4),
                "dti_penalty": round(dti_penalty, 4),
            },
        }

        return risk_level, round(total_risk, 2), 0.7, score_breakdown
    
    def _interpret_prediction(self, prediction: float) -> Tuple[str, float]:
        risk_score = float(prediction * 100)
        if risk_score < 40:
            risk_level = "Low"
        elif risk_score < 60:
            risk_level = "Medium"
        else:
            risk_level = "High"
        return risk_level, risk_score

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
            return self._build_deterministic_override("Reject", ["Low Credit Score"], "Credit score below 550 is an automatic reject.")
        if calculated_ltv > 120:
            return self._build_deterministic_override("Reject", ["High LTV"], "LTV exceeds 120% limit for standard underwriting.")
        if borrower.foir * 100 > 55:
            return self._build_deterministic_override("Reject", ["Extreme FOIR"], "FOIR exceeds 55%, violating ability-to-repay regulations.")
        if borrower.monthly_income <= 0:
            return self._build_deterministic_override("Manual Review", ["Zero Income"], "Income reported as 0 requires manual verification.")

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
            
        category, _ = self._interpret_prediction(final_score / 100.0)

        flag = "Low"
        if models_active > 1:
            scores_list = list(successful_models.values())
            disagreement = (max(scores_list) - min(scores_list)) > 40.0
            if disagreement:
                flag = "High"
                base_confidence -= 0.15
            
        prediction_confidence = round(max(0.1, base_confidence), 2)

        return {
            "ml_ensemble_score": round(final_score / 100.0, 2), # normalized to 0-1
            "override_triggered": False,
            "critical_flags": [],
            "risk_level": category,
            "prediction_confidence": prediction_confidence,
            "fallback_level_used": fallback_level,
            "ensemble_health": ensemble_health,
            "disagreement_flag": flag,
            "models": models_output,
            "recommendation": "Pending AI Review",
            "reasoning": "Ensemble execution completed."
        }

    def _execute_level_4_fallback(self, borrower: BorrowerProfile, models_output: dict, reason: str) -> Dict[str, Any]:
        """Execute Level 4 deterministic fallback when ML ensemble completely fails."""
        risk_level, total_risk, _, _ = self._predict_fallback(borrower)
        if not risk_level.endswith("Risk"):
            risk_level += " Risk"
            
        return {
            "ml_ensemble_score": round(total_risk / 100.0, 2),
            "override_triggered": False,
            "critical_flags": ["Model Inference Failed"],
            "risk_level": risk_level,
            "prediction_confidence": 0.20,
            "fallback_level_used": 4,
            "ensemble_health": "failed",
            "disagreement_flag": "N/A",
            "models": models_output,
            "recommendation": "Manual Review",
            "reasoning": f"ML pipeline unavailable ({reason}). Rule-based assessment used."
        }
        
    def _build_deterministic_override(self, recommendation: str, flags: List[str], reasoning: str) -> Dict[str, Any]:
        """Bypass ML to enforce deterministic banking limits."""
        return {
            "ml_ensemble_score": 1.0 if recommendation == "Reject" else 0.5,
            "override_triggered": True,
            "critical_flags": flags,
            "risk_level": "High Risk",
            "prediction_confidence": 1.0, # Deterministic rules are 100% confident
            "fallback_level_used": 3,
            "ensemble_health": "healthy", # Skipped intentionally, but engine is healthy
            "disagreement_flag": "N/A",
            "models": {},
            "recommendation": recommendation,
            "reasoning": reasoning
        }

# Global singleton
ml_service = MLService()
