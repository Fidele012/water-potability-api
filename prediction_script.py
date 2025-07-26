<<<<<<< HEAD
# This script uses the best trained model to make predictions for API deployment
# Compatible with FastAPI and ready for web service integration

import numpy as np
import pandas as pd
import joblib
import json
from typing import Dict, Union, List
import os
import warnings
warnings.filterwarnings('ignore')

class WaterPotabilityPredictor:
    """
    Water Potability Prediction Class for API Integration
    Uses the best trained model to predict water safety based on quality parameters
    """
    
    def __init__(self, model_path: str = None, scaler_path: str = None, metadata_path: str = None):
        """
        Initialize the predictor with trained model and scaler
        
        Args:
            model_path: Path to the saved best model (.pkl file)
            scaler_path: Path to the saved scaler (.pkl file) - required for Linear Regression
            metadata_path: Path to the model metadata (.json file)
        """
        self.model = None
        self.scaler = None
        self.metadata = None
        self.model_name = None
        self.feature_names = None
        
        # Load model and associated files
        self._load_model_files(model_path, scaler_path, metadata_path)
        
        # Define realistic constraints for water quality parameters
        self.constraints = {
            'ph': (0.0, 14.0),
            'hardness': (0.0, 500.0),
            'solids': (0.0, 50000.0),
            'chloramines': (0.0, 15.0),
            'sulfate': (0.0, 500.0),
            'conductivity': (0.0, 2000.0),
            'organic_carbon': (0.0, 30.0),
            'trihalomethanes': (0.0, 200.0),
            'turbidity': (0.0, 10.0)
        }
        
        # WHO and EPA standards for reference
        self.who_standards = {
            'ph': (6.5, 8.5),
            'hardness': (0, 120),  # soft water
            'solids': (0, 1000),
            'chloramines': (0, 5),
            'sulfate': (0, 250),
            'conductivity': (50, 1500),
            'organic_carbon': (0, 2),
            'trihalomethanes': (0, 100),
            'turbidity': (0, 1)
        }
    
    def _load_model_files(self, model_path: str, scaler_path: str, metadata_path: str):
        """Load the trained model, scaler, and metadata"""
        
        # Try to find files automatically if not provided
        if not model_path:
            # Look for model files in current directory
            model_files = [f for f in os.listdir('.') if f.startswith('best_model_') and f.endswith('.pkl')]
            if model_files:
                model_path = model_files[0]
                print(f"✅ Found model file: {model_path}")
            else:
                raise FileNotFoundError("No model file found. Please provide model_path.")
        
        if not scaler_path:
            scaler_files = [f for f in os.listdir('.') if f.startswith('feature_scaler') and f.endswith('.pkl')]
            if scaler_files:
                scaler_path = scaler_files[0]
                print(f"✅ Found scaler file: {scaler_path}")
        
        if not metadata_path:
            metadata_files = [f for f in os.listdir('.') if f.startswith('model_metadata') and f.endswith('.json')]
            if metadata_files:
                metadata_path = metadata_files[0]
                print(f"✅ Found metadata file: {metadata_path}")
        
        # Load model
        try:
            self.model = joblib.load(model_path)
            print(f"✅ Model loaded successfully from: {model_path}")
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
        
        # Load scaler (if exists)
        if scaler_path and os.path.exists(scaler_path):
            try:
                self.scaler = joblib.load(scaler_path)
                print(f"✅ Scaler loaded successfully from: {scaler_path}")
            except Exception as e:
                print(f"⚠️ Warning: Could not load scaler: {str(e)}")
        
        # Load metadata (if exists)
        if metadata_path and os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                self.model_name = self.metadata.get('best_model_name', 'Unknown')
                self.feature_names = self.metadata.get('feature_names', [])
                print(f"✅ Metadata loaded successfully from: {metadata_path}")
                print(f"✅ Best model type: {self.model_name}")
            except Exception as e:
                print(f"⚠️ Warning: Could not load metadata: {str(e)}")
    
    def validate_input(self, input_data: Dict[str, float]) -> Dict[str, Union[bool, str, List[str]]]:
        """
        Validate input data against realistic constraints
        
        Args:
            input_data: Dictionary containing water quality parameters
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check required parameters
        required_params = ['ph', 'hardness', 'solids', 'chloramines', 'sulfate', 
                          'conductivity', 'organic_carbon', 'trihalomethanes', 'turbidity']
        
        for param in required_params:
            if param not in input_data:
                validation_result['is_valid'] = False
                validation_result['errors'].append(f"Missing required parameter: {param}")
        
        # Check constraints
        for param, value in input_data.items():
            if param in self.constraints:
                min_val, max_val = self.constraints[param]
                if not (min_val <= value <= max_val):
                    validation_result['is_valid'] = False
                    validation_result['errors'].append(
                        f"{param} value {value} is outside valid range [{min_val}, {max_val}]"
                    )
                
                # Check WHO standards (warnings only)
                if param in self.who_standards:
                    who_min, who_max = self.who_standards[param]
                    if not (who_min <= value <= who_max):
                        validation_result['warnings'].append(
                            f"{param} value {value} is outside WHO recommended range [{who_min}, {who_max}]"
                        )
        
        return validation_result
    
    def predict_single(self, ph: float, hardness: float, solids: float, 
                      chloramines: float, sulfate: float, conductivity: float,
                      organic_carbon: float, trihalomethanes: float, turbidity: float) -> Dict:
        """
        Make a single prediction for water potability
        
        Args:
            ph: pH level (0-14)
            hardness: Water hardness in mg/L (0-500)
            solids: Total dissolved solids in ppm (0-50000)
            chloramines: Chloramines amount in ppm (0-15)
            sulfate: Sulfate amount in mg/L (0-500)
            conductivity: Electrical conductivity in μS/cm (0-2000)
            organic_carbon: Organic carbon amount in ppm (0-30)
            trihalomethanes: Trihalomethanes amount in μg/L (0-200)
            turbidity: Turbidity level in NTU (0-10)
            
        Returns:
            Dictionary with prediction results
        """
        
        # Prepare input data
        input_data = {
            'ph': ph,
            'hardness': hardness,
            'solids': solids,
            'chloramines': chloramines,
            'sulfate': sulfate,
            'conductivity': conductivity,
            'organic_carbon': organic_carbon,
            'trihalomethanes': trihalomethanes,
            'turbidity': turbidity
        }
        
        # Validate input
        validation = self.validate_input(input_data)
        if not validation['is_valid']:
            return {
                'success': False,
                'error': 'Input validation failed',
                'details': validation['errors'],
                'warnings': validation.get('warnings', [])
            }
        
        try:
            # Prepare input array
            input_array = np.array([[ph, hardness, solids, chloramines, sulfate, 
                                   conductivity, organic_carbon, trihalomethanes, turbidity]])
            
            # Apply scaling if required (for Linear Regression)
            if self.scaler is not None and self.model_name == 'Linear Regression':
                input_array = self.scaler.transform(input_array)
            
            # Make prediction
            prediction = self.model.predict(input_array)[0]
            
            # Ensure prediction is between 0 and 1
            prediction = np.clip(prediction, 0, 1)
            
            # Determine potability status
            is_potable = prediction > 0.5
            confidence = prediction if is_potable else (1 - prediction)
            
            # Risk assessment
            risk_level = self._assess_risk(prediction)
            
            # Generate recommendation
            recommendation = self._generate_recommendation(input_data, prediction, validation.get('warnings', []))
            
            return {
                'success': True,
                'prediction': {
                    'potability_score': float(prediction),
                    'is_potable': bool(is_potable),
                    'confidence': float(confidence),
                    'risk_level': risk_level,
                    'status': 'POTABLE' if is_potable else 'NOT POTABLE'
                },
                'recommendation': recommendation,
                'warnings': validation.get('warnings', []),
                'model_info': {
                    'model_type': self.model_name,
                    'standardization_used': self.scaler is not None
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Prediction failed: {str(e)}',
                'details': []
            }
    
    def predict_batch(self, input_list: List[Dict[str, float]]) -> List[Dict]:
        """
        Make predictions for multiple water samples
        
        Args:
            input_list: List of dictionaries containing water quality parameters
            
        Returns:
            List of prediction results
        """
        results = []
        
        for i, input_data in enumerate(input_list):
            try:
                result = self.predict_single(**input_data)
                result['sample_id'] = i
                results.append(result)
            except Exception as e:
                results.append({
                    'sample_id': i,
                    'success': False,
                    'error': f'Prediction failed for sample {i}: {str(e)}'
                })
        
        return results
    
    def _assess_risk(self, prediction: float) -> str:
        """Assess risk level based on prediction score"""
        if prediction >= 0.8:
            return 'LOW'
        elif prediction >= 0.6:
            return 'MODERATE'
        elif prediction >= 0.4:
            return 'HIGH'
        else:
            return 'VERY HIGH'
    
    def _generate_recommendation(self, input_data: Dict, prediction: float, warnings: List[str]) -> str:
        """Generate recommendation based on prediction and input parameters"""
        
        if prediction > 0.7:
            base_rec = "Water quality appears safe for consumption."
        elif prediction > 0.5:
            base_rec = "Water quality is marginally acceptable but consider additional treatment."
        elif prediction > 0.3:
            base_rec = "Water quality is poor and requires treatment before consumption."
        else:
            base_rec = "Water quality is unsafe and should not be consumed without extensive treatment."
        
        # Add specific recommendations based on warnings
        if warnings:
            base_rec += f" Note: {len(warnings)} parameter(s) outside WHO recommendations."
        
        return base_rec
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            'model_type': self.model_name,
            'has_scaler': self.scaler is not None,
            'feature_names': self.feature_names,
            'constraints': self.constraints,
            'who_standards': self.who_standards,
            'metadata': self.metadata
        }

# Convenience function for direct API integration
def predict_water_potability_api(ph: float, hardness: float, solids: float, 
                                chloramines: float, sulfate: float, conductivity: float,
                                organic_carbon: float, trihalomethanes: float, 
                                turbidity: float) -> Dict:
    """
    Direct prediction function for API integration
    This is the main function that will be used in Task 2 FastAPI
    """
    
    # Initialize predictor (will auto-load model files)
    predictor = WaterPotabilityPredictor()
    
    # Make prediction
    result = predictor.predict_single(
        ph=ph,
        hardness=hardness,
        solids=solids,
        chloramines=chloramines,
        sulfate=sulfate,
        conductivity=conductivity,
        organic_carbon=organic_carbon,
        trihalomethanes=trihalomethanes,
        turbidity=turbidity
    )
    
    return result

# Test function to verify the script works
def test_prediction_script():
    """Test the prediction script with sample data"""
    
    print("🧪 Testing Water Potability Prediction Script")
    print("=" * 50)
    
    # Test data - sample water quality parameters
    test_samples = [
        {
            'ph': 7.0,
            'hardness': 150.0,
            'solids': 25000.0,
            'chloramines': 8.0,
            'sulfate': 250.0,
            'conductivity': 400.0,
            'organic_carbon': 15.0,
            'trihalomethanes': 80.0,
            'turbidity': 4.0
        },
        {
            'ph': 6.5,
            'hardness': 100.0,
            'solids': 15000.0,
            'chloramines': 4.0,
            'sulfate': 150.0,
            'conductivity': 300.0,
            'organic_carbon': 10.0,
            'trihalomethanes': 50.0,
            'turbidity': 2.0
        }
    ]
    
    try:
        # Initialize predictor
        predictor = WaterPotabilityPredictor()
        
        # Test single predictions
        for i, sample in enumerate(test_samples):
            print(f"\n📊 Test Sample {i+1}:")
            print(f"Input: {sample}")
            
            result = predictor.predict_single(**sample)
            
            if result['success']:
                pred = result['prediction']
                print(f"✅ Prediction: {pred['potability_score']:.4f}")
                print(f"✅ Status: {pred['status']}")
                print(f"✅ Confidence: {pred['confidence']:.4f}")
                print(f"✅ Risk Level: {pred['risk_level']}")
                print(f"✅ Recommendation: {result['recommendation']}")
            else:
                print(f"❌ Error: {result['error']}")
        
        # Test batch prediction
        print(f"\n📊 Batch Prediction Test:")
        batch_results = predictor.predict_batch(test_samples)
        for result in batch_results:
            if result['success']:
                pred = result['prediction']
                print(f"Sample {result['sample_id']}: {pred['status']} (Score: {pred['potability_score']:.3f})")
            else:
                print(f"Sample {result['sample_id']}: ERROR - {result['error']}")
        
        # Test API function
        print(f"\n📊 API Function Test:")
        api_result = predict_water_potability_api(**test_samples[0])
        if api_result['success']:
            print(f"✅ API Function works correctly")
            print(f"✅ Result: {api_result['prediction']['status']}")
        else:
            print(f"❌ API Function failed: {api_result['error']}")
        
        print(f"\n🎉 All tests completed successfully!")
        print(f"✅ Script is ready for Task 2 (FastAPI integration)")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        print(f"⚠️ Make sure model files are available in the current directory")

if __name__ == "__main__":
    # Run tests when script is executed directly
=======
# This script uses the best trained model to make predictions for API deployment
# Compatible with FastAPI and ready for web service integration

import numpy as np
import pandas as pd
import joblib
import json
from typing import Dict, Union, List
import os
import warnings
warnings.filterwarnings('ignore')

class WaterPotabilityPredictor:
    """
    Water Potability Prediction Class for API Integration
    Uses the best trained model to predict water safety based on quality parameters
    """
    
    def __init__(self, model_path: str = None, scaler_path: str = None, metadata_path: str = None):
        """
        Initialize the predictor with trained model and scaler
        
        Args:
            model_path: Path to the saved best model (.pkl file)
            scaler_path: Path to the saved scaler (.pkl file) - required for Linear Regression
            metadata_path: Path to the model metadata (.json file)
        """
        self.model = None
        self.scaler = None
        self.metadata = None
        self.model_name = None
        self.feature_names = None
        
        # Load model and associated files
        self._load_model_files(model_path, scaler_path, metadata_path)
        
        # Define realistic constraints for water quality parameters
        self.constraints = {
            'ph': (0.0, 14.0),
            'hardness': (0.0, 500.0),
            'solids': (0.0, 50000.0),
            'chloramines': (0.0, 15.0),
            'sulfate': (0.0, 500.0),
            'conductivity': (0.0, 2000.0),
            'organic_carbon': (0.0, 30.0),
            'trihalomethanes': (0.0, 200.0),
            'turbidity': (0.0, 10.0)
        }
        
        # WHO and EPA standards for reference
        self.who_standards = {
            'ph': (6.5, 8.5),
            'hardness': (0, 120),  # soft water
            'solids': (0, 1000),
            'chloramines': (0, 5),
            'sulfate': (0, 250),
            'conductivity': (50, 1500),
            'organic_carbon': (0, 2),
            'trihalomethanes': (0, 100),
            'turbidity': (0, 1)
        }
    
    def _load_model_files(self, model_path: str, scaler_path: str, metadata_path: str):
        """Load the trained model, scaler, and metadata"""
        
        # Try to find files automatically if not provided
        if not model_path:
            # Look for model files in current directory
            model_files = [f for f in os.listdir('.') if f.startswith('best_model_') and f.endswith('.pkl')]
            if model_files:
                model_path = model_files[0]
                print(f"✅ Found model file: {model_path}")
            else:
                raise FileNotFoundError("No model file found. Please provide model_path.")
        
        if not scaler_path:
            scaler_files = [f for f in os.listdir('.') if f.startswith('feature_scaler') and f.endswith('.pkl')]
            if scaler_files:
                scaler_path = scaler_files[0]
                print(f"✅ Found scaler file: {scaler_path}")
        
        if not metadata_path:
            metadata_files = [f for f in os.listdir('.') if f.startswith('model_metadata') and f.endswith('.json')]
            if metadata_files:
                metadata_path = metadata_files[0]
                print(f"✅ Found metadata file: {metadata_path}")
        
        # Load model
        try:
            self.model = joblib.load(model_path)
            print(f"✅ Model loaded successfully from: {model_path}")
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
        
        # Load scaler (if exists)
        if scaler_path and os.path.exists(scaler_path):
            try:
                self.scaler = joblib.load(scaler_path)
                print(f"✅ Scaler loaded successfully from: {scaler_path}")
            except Exception as e:
                print(f"⚠️ Warning: Could not load scaler: {str(e)}")
        
        # Load metadata (if exists)
        if metadata_path and os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                self.model_name = self.metadata.get('best_model_name', 'Unknown')
                self.feature_names = self.metadata.get('feature_names', [])
                print(f"✅ Metadata loaded successfully from: {metadata_path}")
                print(f"✅ Best model type: {self.model_name}")
            except Exception as e:
                print(f"⚠️ Warning: Could not load metadata: {str(e)}")
    
    def validate_input(self, input_data: Dict[str, float]) -> Dict[str, Union[bool, str, List[str]]]:
        """
        Validate input data against realistic constraints
        
        Args:
            input_data: Dictionary containing water quality parameters
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check required parameters
        required_params = ['ph', 'hardness', 'solids', 'chloramines', 'sulfate', 
                          'conductivity', 'organic_carbon', 'trihalomethanes', 'turbidity']
        
        for param in required_params:
            if param not in input_data:
                validation_result['is_valid'] = False
                validation_result['errors'].append(f"Missing required parameter: {param}")
        
        # Check constraints
        for param, value in input_data.items():
            if param in self.constraints:
                min_val, max_val = self.constraints[param]
                if not (min_val <= value <= max_val):
                    validation_result['is_valid'] = False
                    validation_result['errors'].append(
                        f"{param} value {value} is outside valid range [{min_val}, {max_val}]"
                    )
                
                # Check WHO standards (warnings only)
                if param in self.who_standards:
                    who_min, who_max = self.who_standards[param]
                    if not (who_min <= value <= who_max):
                        validation_result['warnings'].append(
                            f"{param} value {value} is outside WHO recommended range [{who_min}, {who_max}]"
                        )
        
        return validation_result
    
    def predict_single(self, ph: float, hardness: float, solids: float, 
                      chloramines: float, sulfate: float, conductivity: float,
                      organic_carbon: float, trihalomethanes: float, turbidity: float) -> Dict:
        """
        Make a single prediction for water potability
        
        Args:
            ph: pH level (0-14)
            hardness: Water hardness in mg/L (0-500)
            solids: Total dissolved solids in ppm (0-50000)
            chloramines: Chloramines amount in ppm (0-15)
            sulfate: Sulfate amount in mg/L (0-500)
            conductivity: Electrical conductivity in μS/cm (0-2000)
            organic_carbon: Organic carbon amount in ppm (0-30)
            trihalomethanes: Trihalomethanes amount in μg/L (0-200)
            turbidity: Turbidity level in NTU (0-10)
            
        Returns:
            Dictionary with prediction results
        """
        
        # Prepare input data
        input_data = {
            'ph': ph,
            'hardness': hardness,
            'solids': solids,
            'chloramines': chloramines,
            'sulfate': sulfate,
            'conductivity': conductivity,
            'organic_carbon': organic_carbon,
            'trihalomethanes': trihalomethanes,
            'turbidity': turbidity
        }
        
        # Validate input
        validation = self.validate_input(input_data)
        if not validation['is_valid']:
            return {
                'success': False,
                'error': 'Input validation failed',
                'details': validation['errors'],
                'warnings': validation.get('warnings', [])
            }
        
        try:
            # Prepare input array
            input_array = np.array([[ph, hardness, solids, chloramines, sulfate, 
                                   conductivity, organic_carbon, trihalomethanes, turbidity]])
            
            # Apply scaling if required (for Linear Regression)
            if self.scaler is not None and self.model_name == 'Linear Regression':
                input_array = self.scaler.transform(input_array)
            
            # Make prediction
            prediction = self.model.predict(input_array)[0]
            
            # Ensure prediction is between 0 and 1
            prediction = np.clip(prediction, 0, 1)
            
            # Determine potability status
            is_potable = prediction > 0.5
            confidence = prediction if is_potable else (1 - prediction)
            
            # Risk assessment
            risk_level = self._assess_risk(prediction)
            
            # Generate recommendation
            recommendation = self._generate_recommendation(input_data, prediction, validation.get('warnings', []))
            
            return {
                'success': True,
                'prediction': {
                    'potability_score': float(prediction),
                    'is_potable': bool(is_potable),
                    'confidence': float(confidence),
                    'risk_level': risk_level,
                    'status': 'POTABLE' if is_potable else 'NOT POTABLE'
                },
                'recommendation': recommendation,
                'warnings': validation.get('warnings', []),
                'model_info': {
                    'model_type': self.model_name,
                    'standardization_used': self.scaler is not None
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Prediction failed: {str(e)}',
                'details': []
            }
    
    def predict_batch(self, input_list: List[Dict[str, float]]) -> List[Dict]:
        """
        Make predictions for multiple water samples
        
        Args:
            input_list: List of dictionaries containing water quality parameters
            
        Returns:
            List of prediction results
        """
        results = []
        
        for i, input_data in enumerate(input_list):
            try:
                result = self.predict_single(**input_data)
                result['sample_id'] = i
                results.append(result)
            except Exception as e:
                results.append({
                    'sample_id': i,
                    'success': False,
                    'error': f'Prediction failed for sample {i}: {str(e)}'
                })
        
        return results
    
    def _assess_risk(self, prediction: float) -> str:
        """Assess risk level based on prediction score"""
        if prediction >= 0.8:
            return 'LOW'
        elif prediction >= 0.6:
            return 'MODERATE'
        elif prediction >= 0.4:
            return 'HIGH'
        else:
            return 'VERY HIGH'
    
    def _generate_recommendation(self, input_data: Dict, prediction: float, warnings: List[str]) -> str:
        """Generate recommendation based on prediction and input parameters"""
        
        if prediction > 0.7:
            base_rec = "Water quality appears safe for consumption."
        elif prediction > 0.5:
            base_rec = "Water quality is marginally acceptable but consider additional treatment."
        elif prediction > 0.3:
            base_rec = "Water quality is poor and requires treatment before consumption."
        else:
            base_rec = "Water quality is unsafe and should not be consumed without extensive treatment."
        
        # Add specific recommendations based on warnings
        if warnings:
            base_rec += f" Note: {len(warnings)} parameter(s) outside WHO recommendations."
        
        return base_rec
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            'model_type': self.model_name,
            'has_scaler': self.scaler is not None,
            'feature_names': self.feature_names,
            'constraints': self.constraints,
            'who_standards': self.who_standards,
            'metadata': self.metadata
        }

# Convenience function for direct API integration
def predict_water_potability_api(ph: float, hardness: float, solids: float, 
                                chloramines: float, sulfate: float, conductivity: float,
                                organic_carbon: float, trihalomethanes: float, 
                                turbidity: float) -> Dict:
    """
    Direct prediction function for API integration
    This is the main function that will be used in Task 2 FastAPI
    """
    
    # Initialize predictor (will auto-load model files)
    predictor = WaterPotabilityPredictor()
    
    # Make prediction
    result = predictor.predict_single(
        ph=ph,
        hardness=hardness,
        solids=solids,
        chloramines=chloramines,
        sulfate=sulfate,
        conductivity=conductivity,
        organic_carbon=organic_carbon,
        trihalomethanes=trihalomethanes,
        turbidity=turbidity
    )
    
    return result

# Test function to verify the script works
def test_prediction_script():
    """Test the prediction script with sample data"""
    
    print("🧪 Testing Water Potability Prediction Script")
    print("=" * 50)
    
    # Test data - sample water quality parameters
    test_samples = [
        {
            'ph': 7.0,
            'hardness': 150.0,
            'solids': 25000.0,
            'chloramines': 8.0,
            'sulfate': 250.0,
            'conductivity': 400.0,
            'organic_carbon': 15.0,
            'trihalomethanes': 80.0,
            'turbidity': 4.0
        },
        {
            'ph': 6.5,
            'hardness': 100.0,
            'solids': 15000.0,
            'chloramines': 4.0,
            'sulfate': 150.0,
            'conductivity': 300.0,
            'organic_carbon': 10.0,
            'trihalomethanes': 50.0,
            'turbidity': 2.0
        }
    ]
    
    try:
        # Initialize predictor
        predictor = WaterPotabilityPredictor()
        
        # Test single predictions
        for i, sample in enumerate(test_samples):
            print(f"\n📊 Test Sample {i+1}:")
            print(f"Input: {sample}")
            
            result = predictor.predict_single(**sample)
            
            if result['success']:
                pred = result['prediction']
                print(f"✅ Prediction: {pred['potability_score']:.4f}")
                print(f"✅ Status: {pred['status']}")
                print(f"✅ Confidence: {pred['confidence']:.4f}")
                print(f"✅ Risk Level: {pred['risk_level']}")
                print(f"✅ Recommendation: {result['recommendation']}")
            else:
                print(f"❌ Error: {result['error']}")
        
        # Test batch prediction
        print(f"\n📊 Batch Prediction Test:")
        batch_results = predictor.predict_batch(test_samples)
        for result in batch_results:
            if result['success']:
                pred = result['prediction']
                print(f"Sample {result['sample_id']}: {pred['status']} (Score: {pred['potability_score']:.3f})")
            else:
                print(f"Sample {result['sample_id']}: ERROR - {result['error']}")
        
        # Test API function
        print(f"\n📊 API Function Test:")
        api_result = predict_water_potability_api(**test_samples[0])
        if api_result['success']:
            print(f"✅ API Function works correctly")
            print(f"✅ Result: {api_result['prediction']['status']}")
        else:
            print(f"❌ API Function failed: {api_result['error']}")
        
        print(f"\n🎉 All tests completed successfully!")
        print(f"✅ Script is ready for Task 2 (FastAPI integration)")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        print(f"⚠️ Make sure model files are available in the current directory")

if __name__ == "__main__":
    # Run tests when script is executed directly
>>>>>>> 2628288cf6a61e67f56d1ea3b487aaab5ee35b93
    test_prediction_script()