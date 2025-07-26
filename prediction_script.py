# Machine Learning Model Loading and Prediction Logic
# This script contains the core ML functionality separated from the FastAPI app

import numpy as np
import pandas as pd
import joblib
import os
import warnings
from typing import Dict, List, Optional, Any

# Suppress scikit-learn warnings for cleaner logs
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")

# Define feature names to match exactly what the trained model expects
FEATURE_NAMES = [
    'Ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
    'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'
]

class WaterPotabilityPredictor:
    """Enhanced Water Potability Prediction Class"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.model_loaded = False
        self.load_model()
    
    def load_model(self):
        """Load the trained model and scaler"""
        try:
            # Try to load the best model
            model_files = [f for f in os.listdir('.') if f.startswith('best_model_') and f.endswith('.pkl')]
            if model_files:
                print(f"✅ Loading model: {model_files[0]}")
                self.model = joblib.load(model_files[0])
                
                # Try to load scaler if available
                scaler_files = [f for f in os.listdir('.') if f.startswith('feature_scaler') and f.endswith('.pkl')]
                if scaler_files:
                    print(f"✅ Loading scaler: {scaler_files[0]}")
                    self.scaler = joblib.load(scaler_files[0])
                
                self.model_loaded = True
                print("✅ Model and scaler loaded successfully")
                
                # Debug: Print model feature names if available
                if hasattr(self.model, 'feature_names_in_'):
                    print(f"🔍 Model expects features: {list(self.model.feature_names_in_)}")
                
            else:
                print("❌ No model file found")
                
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            self.model_loaded = False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if not self.model_loaded or self.model is None:
            return {
                "model_type": "None",
                "model_loaded": False,
                "error": "Model not available"
            }
        
        model_info = {
            "model_type": type(self.model).__name__,
            "model_loaded": True,
            "scaler_available": self.scaler is not None,
            "feature_names": FEATURE_NAMES,
            "n_features": len(FEATURE_NAMES)
        }
        
        # Add actual model feature names if available
        if hasattr(self.model, 'feature_names_in_'):
            model_info['model_feature_names'] = list(self.model.feature_names_in_)
        
        # Add model-specific info if available
        if hasattr(self.model, 'n_estimators'):
            model_info['n_estimators'] = self.model.n_estimators
        if hasattr(self.model, 'max_depth'):
            model_info['max_depth'] = self.model.max_depth
            
        return model_info
    
    def get_risk_level(self, score: float) -> str:
        """Calculate risk level based on prediction score"""
        if score >= 0.8:
            return "LOW"
        elif score >= 0.6:
            return "MODERATE" 
        elif score >= 0.4:
            return "HIGH"
        else:
            return "VERY_HIGH"
    
    def generate_recommendations_and_warnings(self, ph, hardness, solids, chloramines, 
                                            sulfate, conductivity, organic_carbon, 
                                            trihalomethanes, turbidity) -> tuple:
        """Generate recommendations and warnings based on WHO/EPA standards"""
        
        recommendations = []
        warnings_list = []
        
        # Parameter validation with WHO/EPA standards
        param_checks = [
            (ph, 6.5, 8.5, "pH level", "pH adjustment treatment"),
            (solids, 0, 1000, "Total dissolved solids (ppm)", "Reverse osmosis or distillation"),
            (chloramines, 0, 5, "Chloramines level (ppm)", "Activated carbon filtration"),
            (turbidity, 0, 1, "Turbidity (NTU)", "Filtration or coagulation treatment"),
            (trihalomethanes, 0, 100, "Trihalomethanes (μg/L)", "Activated carbon treatment"),
            (sulfate, 0, 250, "Sulfate (mg/L)", "Ion exchange treatment"),
            (hardness, 0, 120, "Water hardness (mg/L)", "Water softening treatment")
        ]
        
        for value, min_val, max_val, param_name, treatment in param_checks:
            if value < min_val or value > max_val:
                warnings_list.append(f"{param_name} ({value:.2f}) is outside optimal range ({min_val}-{max_val})")
                recommendations.append(treatment)
        
        # Remove duplicates from recommendations
        recommendations = list(set(recommendations))
        recommendation_text = "; ".join(recommendations) if recommendations else "Water parameters are within acceptable ranges"
        
        return recommendation_text, warnings_list
    
    def predict(self, ph, hardness, solids, chloramines, sulfate, 
                conductivity, organic_carbon, trihalomethanes, turbidity) -> Dict[str, Any]:
        """Make water potability prediction"""
        
        if not self.model_loaded or self.model is None:
            return {
                'success': False,
                'error': 'Model not loaded or not available',
                'details': ['Model loading failed']
            }
        
        try:
            # Get the exact feature names the model expects
            if hasattr(self.model, 'feature_names_in_'):
                expected_features = list(self.model.feature_names_in_)
                print(f"🔍 Using model feature names: {expected_features}")
            else:
                # Fallback to our defined names
                expected_features = FEATURE_NAMES
                print(f"🔍 Using fallback feature names: {expected_features}")
            
            # Create input array in the same order as expected features
            input_values = [ph, hardness, solids, chloramines, sulfate,
                          conductivity, organic_carbon, trihalomethanes, turbidity]
            
            # Create DataFrame with exact feature names from model
            input_data = pd.DataFrame([input_values], columns=expected_features)
            
            # Apply scaling if available
            if self.scaler is not None:
                # For scaler, we might need to use numpy array
                scaled_data = self.scaler.transform(input_data.values)
                # Create new DataFrame with scaled data but same column names
                input_data = pd.DataFrame(scaled_data, columns=expected_features)
            
            # Make prediction with proper feature names
            prediction = self.model.predict(input_data)[0]
            prediction = np.clip(prediction, 0, 1)
            
            # Generate recommendations and warnings
            recommendation_text, warnings_list = self.generate_recommendations_and_warnings(
                ph, hardness, solids, chloramines, sulfate, 
                conductivity, organic_carbon, trihalomethanes, turbidity
            )
            
            return {
                'success': True,
                'prediction': {
                    'potability_score': float(prediction),
                    'is_potable': bool(prediction > 0.5),
                    'confidence': float(prediction if prediction > 0.5 else 1 - prediction),
                    'risk_level': self.get_risk_level(prediction),
                    'status': 'POTABLE' if prediction > 0.5 else 'NOT POTABLE'
                },
                'recommendation': recommendation_text,
                'warnings': warnings_list,
                'model_info': {
                    'model_type': type(self.model).__name__,
                    'standardization_used': self.scaler is not None,
                    'feature_names_used': True,
                    'expected_features': expected_features
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Prediction failed: {str(e)}',
                'details': ['Model prediction processing failed']
            }

# Global predictor instance
_predictor_instance = None

def get_predictor():
    """Get singleton predictor instance"""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = WaterPotabilityPredictor()
    return _predictor_instance

def predict_water_potability_api(ph, hardness, solids, chloramines, sulfate, 
                                conductivity, organic_carbon, trihalomethanes, turbidity):
    """
    Main API function for water potability prediction
    This function is called by the FastAPI endpoints
    """
    
    predictor = get_predictor()
    return predictor.predict(
        ph, hardness, solids, chloramines, sulfate,
        conductivity, organic_carbon, trihalomethanes, turbidity
    )

# Backward compatibility - expose WaterPotabilityPredictor class
# This allows the main API to instantiate the class if needed
__all__ = ['predict_water_potability_api', 'WaterPotabilityPredictor', 'FEATURE_NAMES']
