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
            result["is_list"] = isinstance(data, list)
            result["model_count"] = len(data) if isinstance(data, list) else 0
            
            # Check if models have expected structure
            if isinstance(data, list) and len(data) > 0:
                first_model = data[0]
                if isinstance(first_model, dict):
                    result["first_model_has_name"] = "name" in first_model
                    result["first_model_name"] = first_model.get("name")
        
        return result
    
    def test_get_embedding_status(self) -> Dict[str, Any]:
        """Test getting embedding generation status"""
        response = self.session.get(f"{self.base_url}/api/embeddings/status")
        
        result = {
            "test_name": "get_embedding_status",
            "status_code": response.status_code,
            "success": response.status_code in [200, 404],  # 404 OK if no embeddings exist
            "endpoint": "/api/embeddings/status"
        }
        
        if response.status_code == 200:
            data = response.json()
            result["data"] = data
            
            # Check status structure
            if isinstance(data, dict):
                result["has_status"] = "status" in data
                result["current_status"] = data.get("status")
        
        return result
    
    def test_generate_embeddings_validation(self) -> Dict[str, Any]:
        """Test embedding generation with invalid data (validation test)"""
        # Test with empty/invalid payload
        invalid_payload = {}
        response = self.session.post(f"{self.base_url}/api/embeddings/generate", json=invalid_payload)
        
        result = {
            "test_name": "generate_embeddings_validation",
            "status_code": response.status_code,
            "success": response.status_code in [400, 422],  # Should fail validation
            "endpoint": "/api/embeddings/generate",
            "expected_validation_error": True
        }
        
        return result
    
    def test_delete_embeddings_get(self) -> Dict[str, Any]:
        """Test delete embeddings endpoint structure (GET to check if it exists)"""
        response = self.session.get(f"{self.base_url}/api/embeddings/delete")
        
        result = {
            "test_name": "delete_embeddings_get",
            "status_code": response.status_code,
            "success": response.status_code in [200, 405],  # 405 = Method Not Allowed is OK
            "endpoint": "/api/embeddings/delete"
        }
        
        return result

    def run_all_tests(self) -> list:
        """Run all embedding tests"""
        tests = [
            self.test_get_available_models,
            self.test_get_embedding_status,
            self.test_generate_embeddings_validation,
            self.test_delete_embeddings_get
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
                elif "current_status" in result:
                    extra_info = f" (status: {result['current_status']})"
                
                print(f"  {status} {result['test_name']}: {result['status_code']}{extra_info}")
            except Exception as e:
                results.append({
                    "test_name": test.__name__,
                    "success": False,
                    "error": str(e)
                })
                print(f"  💥 {test.__name__}: ERROR - {e}")
        
        return results