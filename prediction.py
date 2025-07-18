# This API uses the best-performing model from Task 1 to make water potability predictions
# Ready for deployment on Render with public URL and Swagger UI

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
import joblib
import json
import os
import uvicorn
from datetime import datetime

# Import the prediction function from our script
try:
    from prediction_script import predict_water_potability_api, WaterPotabilityPredictor
    print("✅ Prediction script imported successfully")
except ImportError:
    print("⚠️ Warning: prediction_script.py not found. Using fallback prediction function.")
    
    # Fallback prediction function if script is not available
    def predict_water_potability_api(ph, hardness, solids, chloramines, sulfate, 
                                    conductivity, organic_carbon, trihalomethanes, turbidity):
        # Load model and make prediction (basic implementation)
        try:
            # Try to load the best model
            model_files = [f for f in os.listdir('.') if f.startswith('best_model_') and f.endswith('.pkl')]
            if model_files:
                model = joblib.load(model_files[0])
                
                # Try to load scaler if available
                scaler = None
                scaler_files = [f for f in os.listdir('.') if f.startswith('feature_scaler') and f.endswith('.pkl')]
                if scaler_files:
                    scaler = joblib.load(scaler_files[0])
                
                # Prepare input
                input_data = np.array([[ph, hardness, solids, chloramines, sulfate, 
                                      conductivity, organic_carbon, trihalomethanes, turbidity]])
                
                # Apply scaling if available
                if scaler is not None:
                    input_data = scaler.transform(input_data)
                
                # Make prediction
                prediction = model.predict(input_data)[0]
                prediction = np.clip(prediction, 0, 1)
                
                return {
                    'success': True,
                    'prediction': {
                        'potability_score': float(prediction),
                        'is_potable': bool(prediction > 0.5),
                        'confidence': float(prediction if prediction > 0.5 else 1 - prediction),
                        'status': 'POTABLE' if prediction > 0.5 else 'NOT POTABLE'
                    }
                }
            else:
                raise Exception("No model file found")
                
        except Exception as e:
            return {
                'success': False,
                'error': f'Prediction failed: {str(e)}'
            }

# Initialize FastAPI app
app = FastAPI(
    title="Water Potability Prediction API",
    description="AI-powered water quality assessment API using machine learning to predict water potability based on chemical and physical parameters",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI URL
    redoc_url="/redoc"
)

# Add CORS middleware (MANDATORY REQUIREMENT)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Pydantic BaseModel for input validation with data types and range constraints
class WaterQualityInput(BaseModel):
    """
    Water Quality Input Model with enforced data types and range constraints
    All parameters are based on WHO and EPA standards for water quality
    """
    
    ph: float = Field(
        ...,
        ge=0.0,
        le=14.0,
        description="pH level of water (0-14, optimal: 6.5-8.5)"
    )
    
    hardness: float = Field(
        ...,
        ge=0.0,
        le=500.0,
        description="Water hardness in mg/L (0-500, soft: <60, hard: >120)"
    )
    
    solids: float = Field(
        ...,
        ge=0.0,
        le=50000.0,
        description="Total dissolved solids in ppm (0-50000, WHO limit: <1000)"
    )
    
    chloramines: float = Field(
        ...,
        ge=0.0,
        le=15.0,
        description="Chloramines amount in ppm (0-15, WHO limit: <5)"
    )
    
    sulfate: float = Field(
        ...,
        ge=0.0,
        le=500.0,
        description="Sulfate amount in mg/L (0-500, WHO limit: <250)"
    )
    
    conductivity: float = Field(
        ...,
        ge=0.0,
        le=2000.0,
        description="Electrical conductivity in μS/cm (0-2000, typical: 50-1500)"
    )
    
    organic_carbon: float = Field(
        ...,
        ge=0.0,
        le=30.0,
        description="Organic carbon amount in ppm (0-30, typical: <2)"
    )
    
    trihalomethanes: float = Field(
        ...,
        ge=0.0,
        le=200.0,
        description="Trihalomethanes amount in μg/L (0-200, WHO limit: <100)"
    )
    
    turbidity: float = Field(
        ...,
        ge=0.0,
        le=10.0,
        description="Turbidity level in NTU (0-10, WHO limit: <1)"
    )
    
    # Custom validators for additional constraints
    @validator('ph')
    def validate_ph(cls, v):
        if not (0.0 <= v <= 14.0):
            raise ValueError('pH must be between 0 and 14')
        return v
    
    @validator('hardness')
    def validate_hardness(cls, v):
        if v < 0:
            raise ValueError('Hardness cannot be negative')
        return v
    
    @validator('solids')
    def validate_solids(cls, v):
        if v < 0:
            raise ValueError('Total dissolved solids cannot be negative')
        return v
    
    @validator('chloramines')
    def validate_chloramines(cls, v):
        if v < 0:
            raise ValueError('Chloramines cannot be negative')
        return v
    
    @validator('sulfate')
    def validate_sulfate(cls, v):
        if v < 0:
            raise ValueError('Sulfate cannot be negative')
        return v
    
    @validator('conductivity')
    def validate_conductivity(cls, v):
        if v < 0:
            raise ValueError('Conductivity cannot be negative')
        return v
    
    @validator('organic_carbon')
    def validate_organic_carbon(cls, v):
        if v < 0:
            raise ValueError('Organic carbon cannot be negative')
        return v
    
    @validator('trihalomethanes')
    def validate_trihalomethanes(cls, v):
        if v < 0:
            raise ValueError('Trihalomethanes cannot be negative')
        return v
    
    @validator('turbidity')
    def validate_turbidity(cls, v):
        if v < 0:
            raise ValueError('Turbidity cannot be negative')
        return v

# Response model for API output
class PredictionResponse(BaseModel):
    """Response model for prediction results"""
    
    success: bool
    prediction: Optional[Dict[str, Any]] = None
    recommendation: Optional[str] = None
    warnings: Optional[list] = None
    model_info: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    details: Optional[list] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

# Health check endpoint
@app.get("/")
async def root():
    """Root endpoint for health check"""
    return {
        "message": "Water Potability Prediction API",
        "status": "active",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_check": "OK"
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Water Potability Prediction API"
    }

# Main prediction endpoint (POST request as required)
@app.post("/predict", response_model=PredictionResponse)
async def predict_water_potability(input_data: WaterQualityInput):
    """
    Predict water potability based on water quality parameters
    
    This endpoint uses the best-performing model from Task 1 to predict
    whether water is safe for human consumption based on 9 chemical and
    physical parameters.
    
    Parameters:
    - ph: pH level (0-14)
    - hardness: Water hardness in mg/L
    - solids: Total dissolved solids in ppm
    - chloramines: Chloramines amount in ppm
    - sulfate: Sulfate amount in mg/L
    - conductivity: Electrical conductivity in μS/cm
    - organic_carbon: Organic carbon amount in ppm
    - trihalomethanes: Trihalomethanes amount in μg/L
    - turbidity: Turbidity level in NTU
    
    Returns:
    - prediction: Potability score and classification
    - recommendation: Health-based recommendation
    - warnings: Any parameter warnings
    - model_info: Information about the model used
    """
    
    try:
        # Make prediction using the best model from Task 1
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
                prediction=result['prediction'],
                recommendation=result.get('recommendation', 'No recommendation available'),
                warnings=result.get('warnings', []),
                model_info=result.get('model_info', {})
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Prediction failed: {result.get('error', 'Unknown error')}"
            )
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

# Batch prediction endpoint (bonus feature)
@app.post("/predict/batch")
async def predict_batch(input_list: list[WaterQualityInput]):
    """
    Predict water potability for multiple samples
    
    This endpoint allows batch processing of multiple water samples
    for efficiency in processing large datasets.
    """
    
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

# Model information endpoint
@app.get("/model/info")
async def get_model_info():
    """Get information about the loaded model"""
    
    try:
        # Try to get model info from predictor
        predictor = WaterPotabilityPredictor()
        model_info = predictor.get_model_info()
        
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

# Parameters information endpoint
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

# Example usage endpoint
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

# Main entry point for running the server
if __name__ == "__main__":
    # Configuration for deployment
    port = int(os.environ.get("PORT", 8000))
    
    uvicorn.run(
        "prediction:app",  # module:app
        host="0.0.0.0",
        port=port,
        reload=False,  # Set to False for production
        access_log=True
    )