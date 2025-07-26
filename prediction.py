# This API uses the best-performing model from Task 1 to make water potability predictions
# Ready for deployment on Render with public URL and Swagger UI

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd
import joblib
import json
import os
import uvicorn
import warnings
from datetime import datetime

# Suppress scikit-learn warnings for cleaner logs
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")

# Define feature names for consistency
FEATURE_NAMES = [
    'ph', 'hardness', 'solids', 'chloramines', 'sulfate',
    'conductivity', 'organic_carbon', 'trihalomethanes', 'turbidity'
]

# Global variables to store loaded model and scaler
loaded_model = None
loaded_scaler = None
model_loaded = False

def load_model_once():
    """Load model and scaler once at startup to avoid repeated loading"""
    global loaded_model, loaded_scaler, model_loaded
    
    if model_loaded:
        return loaded_model, loaded_scaler
    
    try:
        # Try to load the best model
        model_files = [f for f in os.listdir('.') if f.startswith('best_model_') and f.endswith('.pkl')]
        if model_files:
            print(f"✅ Loading model: {model_files[0]}")
            loaded_model = joblib.load(model_files[0])
            
            # Try to load scaler if available
            scaler_files = [f for f in os.listdir('.') if f.startswith('feature_scaler') and f.endswith('.pkl')]
            if scaler_files:
                print(f"✅ Loading scaler: {scaler_files[0]}")
                loaded_scaler = joblib.load(scaler_files[0])
            
            model_loaded = True
            print("✅ Model and scaler loaded successfully")
            return loaded_model, loaded_scaler
        else:
            print("❌ No model file found")
            return None, None
            
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return None, None

# Import the prediction function from our script
try:
    from prediction_script import predict_water_potability_api, WaterPotabilityPredictor
    print("✅ Prediction script imported successfully")
except ImportError:
    print("⚠️ Using enhanced fallback prediction function")
    
    # Enhanced fallback prediction function
    def predict_water_potability_api(ph, hardness, solids, chloramines, sulfate, 
                                    conductivity, organic_carbon, trihalomethanes, turbidity):
        """Enhanced fallback prediction function with proper feature handling"""
        
        try:
            model, scaler = load_model_once()
            
            if model is None:
                raise Exception("No model available")
            
            # Create DataFrame with proper feature names to avoid warnings
            input_data = pd.DataFrame({
                'ph': [ph],
                'hardness': [hardness], 
                'solids': [solids],
                'chloramines': [chloramines],
                'sulfate': [sulfate],
                'conductivity': [conductivity],
                'organic_carbon': [organic_carbon],
                'trihalomethanes': [trihalomethanes],
                'turbidity': [turbidity]
            })
            
            # Apply scaling if available
            if scaler is not None:
                # Convert to numpy array for scaler, then back to DataFrame
                scaled_data = scaler.transform(input_data.values)
                input_data = pd.DataFrame(scaled_data, columns=FEATURE_NAMES)
            
            # Make prediction with proper feature names
            prediction = model.predict(input_data)[0]
            prediction = np.clip(prediction, 0, 1)
            
            # Calculate risk level based on prediction score
            def get_risk_level(score):
                if score >= 0.8:
                    return "LOW"
                elif score >= 0.6:
                    return "MODERATE" 
                elif score >= 0.4:
                    return "HIGH"
                else:
                    return "VERY_HIGH"
            
            # Generate recommendations and warnings based on parameters
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
            
            return {
                'success': True,
                'prediction': {
                    'potability_score': float(prediction),
                    'is_potable': bool(prediction > 0.5),
                    'confidence': float(prediction if prediction > 0.5 else 1 - prediction),
                    'risk_level': get_risk_level(prediction),
                    'status': 'POTABLE' if prediction > 0.5 else 'NOT POTABLE'
                },
                'recommendation': recommendation_text,
                'warnings': warnings_list,
                'model_info': {
                    'model_type': type(model).__name__ if model else 'Unknown',
                    'standardization_used': scaler is not None,
                    'feature_names_used': True
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Prediction failed: {str(e)}',
                'details': ['Model loading or prediction processing failed']
            }

# Initialize FastAPI app
app = FastAPI(
    title="Water Potability Prediction API",
    description="AI-powered water quality assessment API using machine learning to predict water potability based on chemical and physical parameters",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
@app.on_event("startup")
async def startup_event():
    """Load model and scaler at application startup"""
    print("🚀 Starting Water Potability Prediction API...")
    load_model_once()
    print("✅ API startup complete")

# Response models
class PredictionOutput(BaseModel):
    potability_score: float = Field(..., ge=0.0, le=1.0, description="Prediction score between 0-1")
    is_potable: bool = Field(..., description="Whether water is safe to drink")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence in prediction")
    risk_level: str = Field(..., description="LOW/MODERATE/HIGH/VERY_HIGH")
    status: str = Field(..., description="POTABLE or NOT POTABLE")

class ModelInfo(BaseModel):
    model_type: str = Field(default="RandomForestRegressor", description="Type of ML model used")
    standardization_used: bool = Field(default=False, description="Whether features were standardized")
    feature_names_used: bool = Field(default=True, description="Whether proper feature names were used")

class PredictionResponse(BaseModel):
    success: bool
    prediction: Optional[PredictionOutput] = None
    recommendation: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)
    model_info: ModelInfo = Field(default_factory=ModelInfo)
    error: Optional[str] = None
    details: Optional[List[str]] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

# Input validation model
class WaterQualityInput(BaseModel):
    """Water Quality Input Model with enforced data types and range constraints"""
    
    ph: float = Field(..., ge=0.0, le=14.0, description="pH level of water (0-14, optimal: 6.5-8.5)")
    hardness: float = Field(..., ge=0.0, le=500.0, description="Water hardness in mg/L (0-500, soft: <60, hard: >120)")
    solids: float = Field(..., ge=0.0, le=50000.0, description="Total dissolved solids in ppm (0-50000, WHO limit: <1000)")
    chloramines: float = Field(..., ge=0.0, le=15.0, description="Chloramines amount in ppm (0-15, WHO limit: <5)")
    sulfate: float = Field(..., ge=0.0, le=500.0, description="Sulfate amount in mg/L (0-500, WHO limit: <250)")
    conductivity: float = Field(..., ge=0.0, le=2000.0, description="Electrical conductivity in μS/cm (0-2000, typical: 50-1500)")
    organic_carbon: float = Field(..., ge=0.0, le=30.0, description="Organic carbon amount in ppm (0-30, typical: <2)")
    trihalomethanes: float = Field(..., ge=0.0, le=200.0, description="Trihalomethanes amount in μg/L (0-200, WHO limit: <100)")
    turbidity: float = Field(..., ge=0.0, le=10.0, description="Turbidity level in NTU (0-10, WHO limit: <1)")
    
    @validator('ph')
    def validate_ph(cls, v):
        if not (0.0 <= v <= 14.0):
            raise ValueError('pH must be between 0 and 14')
        return v
    
    @validator('hardness', 'solids', 'chloramines', 'sulfate', 'conductivity', 'organic_carbon', 'trihalomethanes', 'turbidity')
    def validate_non_negative(cls, v):
        if v < 0:
            raise ValueError('Value cannot be negative')
        return v

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint for health check"""
    return {
        "message": "Water Potability Prediction API",
        "status": "active",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_check": "OK",
        "model_loaded": model_loaded
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Water Potability Prediction API",
        "model_status": "loaded" if model_loaded else "not_loaded"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_water_potability(input_data: WaterQualityInput):
    """Predict water potability based on water quality parameters"""
    
    try:
        # Make prediction using enhanced fallback function
        result = predict_water_potability_api(
            ph=input_data.ph,
            hardness=input_data.hardness,
            solids=input_data.solids,
            chloramines=input_data.chloramines,
            sulfate=input_data.sulfate,
            conductivity=input_data.conductivity,
            organic_carbon=input_data.organic_carbon,
            trihalomethanes=input_data.trihalomethanes,
            turbidity=input_data.turbidity
        )
        
        if result['success']:
            return PredictionResponse(
                success=True,
                prediction=PredictionOutput(**result['prediction']),
                recommendation=result.get('recommendation', 'No recommendation available'),
                warnings=result.get('warnings', []),
                model_info=ModelInfo(**result.get('model_info', {}))
            )
        else:
            return PredictionResponse(
                success=False,
                error=result.get('error', 'Unknown error'),
                details=result.get('details', [])
            )
            
    except Exception as e:
        return PredictionResponse(
            success=False,
            error=f"Internal server error: {str(e)}",
            details=["Prediction processing failed"]
        )

@app.post("/predict/batch")
async def predict_batch(input_list: list[WaterQualityInput]):
    """Predict water potability for multiple samples"""
    
    try:
        results = []
        
        for i, input_data in enumerate(input_list):
            result = predict_water_potability_api(
                ph=input_data.ph,
                hardness=input_data.hardness,
                solids=input_data.solids,
                chloramines=input_data.chloramines,
                sulfate=input_data.sulfate,
                conductivity=input_data.conductivity,
                organic_carbon=input_data.organic_carbon,
                trihalomethanes=input_data.trihalomethanes,
                turbidity=input_data.turbidity
            )
            
            result['sample_id'] = i
            results.append(result)
        
        return {
            "success": True,
            "results": results,
            "total_samples": len(input_list),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )

@app.get("/model/info")
async def get_model_info():
    """Get information about the loaded model"""
    
    try:
        model, scaler = load_model_once()
        
        if model is not None:
            model_info = {
                "model_type": type(model).__name__,
                "model_loaded": True,
                "scaler_available": scaler is not None,
                "feature_names": FEATURE_NAMES,
                "n_features": len(FEATURE_NAMES)
            }
            
            # Add model-specific info if available
            if hasattr(model, 'n_estimators'):
                model_info['n_estimators'] = model.n_estimators
            if hasattr(model, 'max_depth'):
                model_info['max_depth'] = model.max_depth
                
        else:
            model_info = {
                "model_type": "None",
                "model_loaded": False,
                "error": "Model not available"
            }
        
        return {
            "success": True,
            "model_info": model_info,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Could not retrieve model info: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

@app.get("/parameters/info")
async def get_parameters_info():
    """Get information about water quality parameters and their acceptable ranges"""
    
    parameters_info = {
        "ph": {
            "name": "pH Level",
            "unit": "pH scale",
            "range": [0.0, 14.0],
            "optimal": [6.5, 8.5],
            "description": "Measure of acidity or alkalinity of water"
        },
        "hardness": {
            "name": "Water Hardness",
            "unit": "mg/L",
            "range": [0.0, 500.0],
            "optimal": [0, 120],
            "description": "Concentration of calcium and magnesium ions"
        },
        "solids": {
            "name": "Total Dissolved Solids",
            "unit": "ppm",
            "range": [0.0, 50000.0],
            "optimal": [0, 1000],
            "description": "Total amount of dissolved minerals and salts"
        },
        "chloramines": {
            "name": "Chloramines",
            "unit": "ppm",
            "range": [0.0, 15.0],
            "optimal": [0, 5],
            "description": "Chemical compounds used for water disinfection"
        },
        "sulfate": {
            "name": "Sulfate",
            "unit": "mg/L",
            "range": [0.0, 500.0],
            "optimal": [0, 250],
            "description": "Naturally occurring salt in water"
        },
        "conductivity": {
            "name": "Electrical Conductivity",
            "unit": "μS/cm",
            "range": [0.0, 2000.0],
            "optimal": [50, 1500],
            "description": "Ability of water to conduct electric current"
        },
        "organic_carbon": {
            "name": "Organic Carbon",
            "unit": "ppm",
            "range": [0.0, 30.0],
            "optimal": [0, 2],
            "description": "Amount of organic matter in water"
        },
        "trihalomethanes": {
            "name": "Trihalomethanes",
            "unit": "μg/L",
            "range": [0.0, 200.0],
            "optimal": [0, 100],
            "description": "Chemical compounds formed during water treatment"
        },
        "turbidity": {
            "name": "Turbidity",
            "unit": "NTU",
            "range": [0.0, 10.0],
            "optimal": [0, 1],
            "description": "Measure of water clarity"
        }
    }
    
    return {
        "success": True,
        "parameters": parameters_info,
        "note": "Optimal ranges are based on WHO and EPA standards",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/example")
async def get_example():
    """Get example input data for testing the API"""
    
    example_data = {
        "ph": 7.0,
        "hardness": 150.0,
        "solids": 25000.0,
        "chloramines": 8.0,
        "sulfate": 250.0,
        "conductivity": 400.0,
        "organic_carbon": 15.0,
        "trihalomethanes": 80.0,
        "turbidity": 4.0
    }
    
    return {
        "success": True,
        "example_input": example_data,
        "note": "Copy this data to test the /predict endpoint",
        "timestamp": datetime.now().isoformat()
    }

# Main entry point
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    
    uvicorn.run(
        "prediction:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        access_log=True
    )
