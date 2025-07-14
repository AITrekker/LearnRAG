"""
Embeddings endpoint tests
"""
import requests
from typing import Dict, Any

class EmbeddingTests:
    def __init__(self, base_url: str, session: requests.Session):
        self.base_url = base_url
        self.session = session
        
    def test_get_available_models(self) -> Dict[str, Any]:
        """Test getting available embedding models"""
        response = self.session.get(f"{self.base_url}/api/embeddings/models")
        
        result = {
            "test_name": "get_available_models",
            "status_code": response.status_code,
            "success": response.status_code == 200,
            "endpoint": "/api/embeddings/models"
        }
        
        if response.status_code == 200:
            data = response.json()
            result["data"] = data
            # Should have 'models' key with list of models
            result["has_models_key"] = "models" in data
            if "models" in data:
                models = data["models"]
                result["model_count"] = len(models) if isinstance(models, list) else 0
                # Check if we have at least the 5 expected models
                result["has_expected_models"] = result["model_count"] >= 5
                
                # Check first model structure
                if isinstance(models, list) and len(models) > 0:
                    first_model = models[0]
                    required_fields = ["name", "description", "dimension", "default"]
                    result["valid_model_structure"] = all(field in first_model for field in required_fields)
                    result["success"] = result["success"] and result["valid_model_structure"]
        
        return result
    
    def test_get_status_summary(self) -> Dict[str, Any]:
        """Test getting embedding status summary"""
        response = self.session.get(f"{self.base_url}/api/embeddings/status-summary")
        
        result = {
            "test_name": "get_status_summary",
            "status_code": response.status_code,
            "success": response.status_code == 200,
            "endpoint": "/api/embeddings/status-summary"
        }
        
        if response.status_code == 200:
            data = response.json()
            result["data"] = data
            # Check summary structure
            required_fields = ["total_files", "files_with_embeddings", "total_chunks"]
            result["has_required_fields"] = all(field in data for field in required_fields)
            result["success"] = result["success"] and result["has_required_fields"]
        
        return result
    
    def test_generate_embeddings_invalid_payload(self) -> Dict[str, Any]:
        """Test embedding generation with invalid data (validation test)"""
        # Test with empty/invalid payload
        invalid_payload = {}
        response = self.session.post(f"{self.base_url}/api/embeddings/generate", json=invalid_payload)
        
        result = {
            "test_name": "generate_embeddings_invalid_payload",
            "status_code": response.status_code,
            "success": response.status_code in [400, 422],  # Should fail validation
            "endpoint": "/api/embeddings/generate",
            "method": "POST",
            "expected_validation_error": True
        }
        
        return result
    
    def test_get_chunking_strategies(self) -> Dict[str, Any]:
        """Test getting available chunking strategies"""
        response = self.session.get(f"{self.base_url}/api/embeddings/chunking-strategies")
        
        result = {
            "test_name": "get_chunking_strategies",
            "status_code": response.status_code,
            "success": response.status_code == 200,
            "endpoint": "/api/embeddings/chunking-strategies"
        }
        
        if response.status_code == 200:
            data = response.json()
            result["data"] = data
            # Should have 'strategies' key with list of strategies
            result["has_strategies_key"] = "strategies" in data
            if "strategies" in data:
                strategies = data["strategies"]
                result["strategy_count"] = len(strategies) if isinstance(strategies, list) else 0
                # Check if we have the 3 expected strategies
                result["has_expected_strategies"] = result["strategy_count"] >= 3
                
                # Check first strategy structure
                if isinstance(strategies, list) and len(strategies) > 0:
                    first_strategy = strategies[0]
                    required_fields = ["name", "description", "parameters", "default"]
                    result["valid_strategy_structure"] = all(field in first_strategy for field in required_fields)
                    result["success"] = result["success"] and result["valid_strategy_structure"]
        
        return result
    
    def test_get_current_metrics(self) -> Dict[str, Any]:
        """Test getting current embedding generation metrics"""
        response = self.session.get(f"{self.base_url}/api/embeddings/metrics/current")
        
        result = {
            "test_name": "get_current_metrics",
            "status_code": response.status_code,
            "success": response.status_code == 200,
            "endpoint": "/api/embeddings/metrics/current"
        }
        
        if response.status_code == 200:
            data = response.json()
            result["data"] = data
            # Should have 'active' key
            result["has_active_key"] = "active" in data
            result["is_active"] = data.get("active", False)
            # If active, should have progress data
            if data.get("active"):
                result["has_progress"] = "progress" in data
            result["success"] = result["success"] and result["has_active_key"]
        
        return result
    
    def test_generate_embeddings_valid_payload(self) -> Dict[str, Any]:
        """Test embedding generation endpoint with valid data"""
        # Test with minimal valid payload (should work)
        valid_payload = {
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "chunking_strategy": "fixed_size",
            "chunk_size": 512,
            "chunk_overlap": 50
        }
        response = self.session.post(f"{self.base_url}/api/embeddings/generate", json=valid_payload)
        
        result = {
            "test_name": "generate_embeddings_valid_payload",
            "status_code": response.status_code,
            "success": response.status_code == 200,
            "endpoint": "/api/embeddings/generate",
            "method": "POST"
        }
        
        if response.status_code == 200:
            data = response.json()
            result["data"] = data
            # Should have response fields
            required_fields = ["message", "processed_files", "embedding_model", "chunking_strategy"]
            result["has_required_fields"] = all(field in data for field in required_fields)
            result["processed_file_count"] = data.get("processed_files", 0)
            result["success"] = result["success"] and result["has_required_fields"]
        
        return result

    def run_all_tests(self) -> list:
        """Run all embedding tests"""
        tests = [
            self.test_get_available_models,
            self.test_get_status_summary,
            self.test_get_chunking_strategies,
            self.test_get_current_metrics,
            self.test_generate_embeddings_invalid_payload,
            self.test_generate_embeddings_valid_payload
        ]
        
        results = []
        for test in tests:
            try:
                result = test()
                results.append(result)
                status = "✅" if result["success"] else "❌"
                extra_info = ""
                
                # Add useful info for display
                if "model_count" in result:
                    extra_info = f" ({result['model_count']} models)"
                elif "strategy_count" in result:
                    extra_info = f" ({result['strategy_count']} strategies)"
                elif "processed_file_count" in result:
                    extra_info = f" ({result['processed_file_count']} files)"
                elif "is_active" in result and result["is_active"]:
                    extra_info = " (generation active)"
                elif result.get("expected_validation_error"):
                    extra_info = " (validation error expected)"
                
                print(f"  {status} {result['test_name']}: {result['status_code']}{extra_info}")
            except Exception as e:
                results.append({
                    "test_name": test.__name__,
                    "success": False,
                    "error": str(e)
                })
                print(f"  💥 {test.__name__}: ERROR - {e}")
        
        return results