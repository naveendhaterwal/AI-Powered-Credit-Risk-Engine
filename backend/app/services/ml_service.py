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
        self.models_loaded = False
        self.use_fallback = False
        self._load_models()
    
    def _load_models(self):
        """Load multiple ML models from disk"""
        try:
            feature_path = Path(settings.FEATURE_COLUMNS_PATH)
            if not feature_path.exists():
                logger.warning(f"Feature columns not found at {feature_path}. Switching to rule-based fallback.")
                self.use_fallback = True
                return
            
            self.feature_columns = joblib.load(feature_path)
            
            # Map of model name to its expected filename
            model_registry = {
                "logistic": "logistic_model.pkl",
                "random_forest": "rf_model.pkl",
                "gradient_boosting": "gb_model.pkl",
                "xgboost": "xgb_model.pkl"
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
                logger.info(f"ML Service initialized with {loaded_count} models and {len(self.feature_columns)} features")
                
        except Exception as e:
            logger.error(f"Error in ML Service initialization: {str(e)}. Enabling rule-based fallback.")
            self.models = {}
            self.feature_columns = []
            self.models_loaded = False
            self.use_fallback = True

    def _map_age_bucket(self, age: int) -> str:
        """Map numeric age to bucket labels typically used in credit datasets."""
        if age < 25:
            return "<25"
        if age < 35:
            return "25-34"
        if age < 45:
            return "35-44"
        if age < 55:
            return "45-54"
        if age < 65:
            return "55-64"
        if age < 75:
            return "65-74"
        return ">74"

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
        """Build a one-row DataFrame aligned to the trained model feature columns."""
        if not self.feature_columns:
            raise ValueError("Feature columns are not loaded")

        # Start with NaN so the model's imputers can fill missing values.
        row = {col: np.nan for col in self.feature_columns}

        # Numeric values from borrower profile.
        row["income"] = float(borrower.monthly_income * 12)
        row["loan_amount"] = float(borrower.loan_amount_requested)
        row["Credit_Score"] = int(borrower.credit_score)
        row["dtir1"] = float(max(0.0, min(100.0, borrower.dti * 100.0)))
        row["LTV"] = 80.0
        row["term"] = int(borrower.loan_tenure_months)

        # Categorical values mapped to model vocabulary used in training UI.
        row["loan_purpose"] = self._loan_purpose_to_model(borrower.loan_purpose)
        row["Credit_Worthiness"] = "l1" if borrower.credit_score >= 700 else "l2"
        row["open_credit"] = "opc" if borrower.existing_loan_amount > 0 else "nopc"
        row["interest_only"] = "not_int"
        row["loan_limit"] = "cf"
        row["business_or_commercial"] = "ob/c" if borrower.loan_purpose == "Business" else "nob/c"
        row["occupancy_type"] = "pr"
        row["age"] = self._map_age_bucket(borrower.age)

        # Defaults for remaining categorical fields from common mortgage dataset values.
        row["year"] = 2019
        row["Gender"] = "Sex Not Available"
        row["approv_in_adv"] = "nopre"
        row["loan_type"] = "type1"
        row["rate_of_interest"] = 8.5
        row["Interest_rate_spread"] = 0.0
        row["Upfront_charges"] = 0.0
        row["Neg_ammortization"] = "not_neg"
        row["lump_sum_payment"] = "not_lpsm"
        row["property_value"] = float(borrower.loan_amount_requested / 0.8) if borrower.loan_amount_requested > 0 else 0.0
        row["construction_type"] = "sb"
        row["Secured_by"] = "home"
        row["total_units"] = "1U"
        row["credit_type"] = "EXP"
        row["co-applicant_credit_type"] = "CIB"
        row["submission_of_application"] = "to_inst"
        row["Region"] = "south"
        row["Security_Type"] = "direct"

        ordered_row = {col: row.get(col, np.nan) for col in self.feature_columns}
        return pd.DataFrame([ordered_row])
    
    def predict_all_models(self, borrower: BorrowerProfile) -> Dict[str, Any]:
        """
        Predict using all models and return ensemble details.
        """
        if self.use_fallback or not self.models_loaded:
            # Fallback if models are not loaded
            return {
                "logistic": 0.0,
                "random_forest": 0.0,
                "gradient_boost": 0.0,
                "ensemble_score": 0.0,
                "risk_level": "Medium Risk",
                "disagreement_flag": "Low"
            }

        features = self._prepare_features(borrower)
        individual_probs = {}
        for name, model in self.models.items():
            if name in ["logistic", "random_forest", "gradient_boosting"]:
                try:
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(features)[0]
                        pos_idx = 1
                        if hasattr(model, "classes_"):
                            classes = list(model.classes_)
                            if 1 in classes:
                                pos_idx = classes.index(1)
                            elif '1' in classes:
                                pos_idx = classes.index('1')
                        individual_probs[name] = float(probs[pos_idx])
                except Exception as e:
                    logger.error(f"Error predicting with {name}: {e}")

        log_prob = individual_probs.get("logistic", 0.0)
        rf_prob = individual_probs.get("random_forest", 0.0)
        gb_prob = individual_probs.get("gradient_boosting", 0.0)

        final_score = 0.3 * log_prob + 0.35 * rf_prob + 0.35 * gb_prob

        if final_score <= 0.4:
            category = "Low Risk"
        elif final_score <= 0.7:
            category = "Medium Risk"
        else:
            category = "High Risk"

        probs_list = [log_prob, rf_prob, gb_prob]
        flag = "High Disagreement" if (max(probs_list) - min(probs_list)) > 0.4 else "Low Disagreement"

        # To match the requested JSON output format strictly, we use exactly "High" or "Low" 
        # or whatever the flag should be, but let's use "High" / "Low" as it matches the example.
        if flag == "High Disagreement":
            flag = "High"
        else:
            flag = "Low"

        return {
            "logistic": round(log_prob, 2),
            "random_forest": round(rf_prob, 2),
            "gradient_boost": round(gb_prob, 2),
            "ensemble_score": round(final_score, 2),
            "risk_level": category,
            "disagreement_flag": flag
        }

    def _prepare_features(self, borrower: BorrowerProfile) -> pd.DataFrame:
        """Prepare borrower data for the saved sklearn pipeline."""
        return self._build_model_input(borrower)
            
    def _predict_fallback(self, borrower: BorrowerProfile) -> Tuple[str, float, float, Dict[str, Any]]:
        """Rule-based risk calculation as a fallback for missing or incompatible ML model."""
        logger.info("Using rule-based risk prediction fallback")
        
        # Base score from Credit Score (higher = better)
        # Convert 300-900 range to 0-100 (inverted for risk)
        base_risk = 100 - ((borrower.credit_score - 300) / 600 * 100)
        
        # Adjust for FOIR (Banks prefer < 40%)
        # FOIR of 0.5 adds significant risk
        foir_penalty = max(0, (borrower.foir - 0.4) * 100)
        
        # Adjust for DTI (Banks prefer < 40%)
        dti_penalty = max(0, (borrower.dti - 0.4) * 50)
        
        # Employment stability
        employment_bonus = -5 if borrower.employment_type == "Salaried" else 5
        
        total_risk = base_risk + foir_penalty + dti_penalty + employment_bonus
        total_risk = max(1.0, min(100, total_risk))
        
        risk_level, _ = self._interpret_prediction(total_risk / 100.0)
        
        score_breakdown = {
            "method": "rule_based_fallback",
            "formula": "risk = clamp(base_risk + foir_penalty + dti_penalty + employment_adjustment, 0, 100)",
            "strict_no_fallbacks": settings.STRICT_NO_FALLBACKS,
            "components": {
                "base_risk_from_credit_score": round(base_risk, 4),
                "foir_penalty": round(foir_penalty, 4),
                "dti_penalty": round(dti_penalty, 4),
                "employment_adjustment": employment_bonus,
            },
        }

        return risk_level, round(total_risk, 2), 0.7, score_breakdown  # Constant confidence for fallback
    
    def _interpret_prediction(self, prediction: float) -> Tuple[str, float]:
        """
        Convert raw model prediction to risk level and score.
        
        Assumes prediction is between 0-1 where:
        - 0 = Low risk
        - 1 = High risk
        """
        
        # Convert to 0-100 scale
        risk_score = float(prediction * 100)
        
        # Classify into levels
        if risk_score < 40:
            risk_level = "Low"
        elif risk_score < 60:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        return risk_level, risk_score


# Global instance - created once and reused
ml_service = MLService()
