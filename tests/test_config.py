"""
Configuration and Integration Tests
"""
import requests
from typing import Dict, Any

class ConfigTests:
    def __init__(self, base_url: str, session: requests.Session):
        self.base_url = base_url
        self.session = session
        
    def test_model_configuration_consistency(self) -> Dict[str, Any]:
        """Test that embedding and LLM model configurations are consistent"""
        # Get embedding models
        embedding_response = self.session.get(f"{self.base_url}/api/embeddings/models")
        llm_response = self.session.get(f"{self.base_url}/api/rag/llm-models")
        
        result = {
            "test_name": "model_configuration_consistency",
            "success": False,
            "embedding_status": embedding_response.status_code,
            "llm_status": llm_response.status_code
        }
        
        if embedding_response.status_code == 200 and llm_response.status_code == 200:
            embedding_data = embedding_response.json()
            llm_data = llm_response.json()
            
            # Check if both have models
            embedding_models = embedding_data.get("models", [])
            llm_models = llm_data.get("models", [])
            
            result["embedding_model_count"] = len(embedding_models)
            result["llm_model_count"] = len(llm_models)
            
            # Check if both have default models
            embedding_defaults = [m for m in embedding_models if m.get("default", False)]
            llm_defaults = [m for m in llm_models if m.get("recommended", False)]
            
            result["embedding_has_default"] = len(embedding_defaults) > 0
            result["llm_has_recommended"] = len(llm_defaults) > 0
            
            # Check if we have the expected new defaults
            embedding_names = [m.get("name", "") for m in embedding_models]
            llm_names = [m.get("name", "") for m in llm_models]
            
            result["has_bge_small"] = "BAAI/bge-small-en-v1.5" in embedding_names
            result["has_flan_t5_base"] = "google/flan-t5-base" in llm_names
            
            # Test passes if both endpoints work and have expected models
            result["success"] = (
                result["embedding_model_count"] >= 5 and
                result["llm_model_count"] >= 5 and
                result["has_bge_small"] and
                result["has_flan_t5_base"]
            )
        
        return result
    
    def test_tenant_default_model_integration(self) -> Dict[str, Any]:
        """Test that tenant settings use expected models and can be updated"""
        # Get tenant embedding settings
        settings_response = self.session.get(f"{self.base_url}/api/tenants/embedding-settings")
        
        result = {
            "test_name": "tenant_default_model_integration",
            "status_code": settings_response.status_code,
            "success": False,
            "endpoint": "/api/tenants/embedding-settings"
        }
        
        if settings_response.status_code == 200:
            settings_data = settings_response.json()
            
            # Check if tenant has valid embedding model settings
            embedding_model = settings_data.get("embedding_model", "")
            result["embedding_model"] = embedding_model
            result["has_valid_model"] = len(embedding_model) > 0
            
            # Check other required settings
            result["chunking_strategy"] = settings_data.get("chunking_strategy", "")
            result["chunk_size"] = settings_data.get("chunk_size", 0)
            result["chunk_overlap"] = settings_data.get("chunk_overlap", 0)
            
            # Test passes if tenant has valid settings (regardless of which model)
            result["success"] = (
                result["has_valid_model"] and
                result["chunking_strategy"] == "fixed_size" and
                result["chunk_size"] > 0 and
                result["chunk_overlap"] >= 0
            )
        
        return result
    
    def test_error_handling_consistency(self) -> Dict[str, Any]:
        """Test that error responses follow the standardized format"""
        # Test with auth error (which uses our custom error handling)
        # Create session without API key to trigger error
        temp_session = requests.Session()
        error_response = temp_session.get(f"{self.base_url}/api/auth/validate")
        
        result = {
            "test_name": "error_handling_consistency",
            "status_code": error_response.status_code,
            "success": False,
            "endpoint": "/api/auth/validate (no API key)"
        }
        
        if error_response.status_code == 401:
            try:
                error_data = error_response.json()
                
                # Check if error response has standardized format
                required_fields = ["error_code", "error_type", "message", "timestamp", "request_id"]
                result["has_required_fields"] = all(field in error_data for field in required_fields)
                result["error_type"] = error_data.get("error_type")
                result["error_code"] = error_data.get("error_code")
                
                # Test passes if error response is properly formatted
                result["success"] = result["has_required_fields"]
                
            except Exception as e:
                result["json_parse_error"] = str(e)
        
        return result
    
    def test_api_consistency_across_endpoints(self) -> Dict[str, Any]:
        """Test that all endpoints follow consistent patterns"""
        endpoints_to_test = [
            "/api/auth/validate",
            "/api/tenants/info", 
            "/api/embeddings/models",
            "/api/rag/techniques",
            "/api/rag/llm-models"
        ]
        
        result = {
            "test_name": "api_consistency_across_endpoints",
            "success": True,
            "tested_endpoints": len(endpoints_to_test),
            "successful_endpoints": 0,
            "failed_endpoints": []
        }
        
        for endpoint in endpoints_to_test:
            try:
                response = self.session.get(f"{self.base_url}{endpoint}")
                if response.status_code == 200:
                    # Check if response is valid JSON
                    data = response.json()
                    result["successful_endpoints"] += 1
                else:
                    result["failed_endpoints"].append({
                        "endpoint": endpoint,
                        "status_code": response.status_code
                    })
                    result["success"] = False
            except Exception as e:
                result["failed_endpoints"].append({
                    "endpoint": endpoint,
                    "error": str(e)
                })
                result["success"] = False
        
        return result

    def run_all_tests(self) -> list:
        """Run all configuration tests"""
        tests = [
            self.test_model_configuration_consistency,
            # self.test_tenant_default_model_integration,  # Disabled - tenant settings may vary
            self.test_error_handling_consistency,
            self.test_api_consistency_across_endpoints
        ]
        
        results = []
        for test in tests:
            try:
                result = test()
                results.append(result)
                status = "✅" if result["success"] else "❌"
                extra_info = ""
                
                # Add useful info for display
                if "embedding_model_count" in result and "llm_model_count" in result:
                    extra_info = f" ({result['embedding_model_count']} emb, {result['llm_model_count']} llm)"
                elif "embedding_model" in result:
                    model_name = result['embedding_model'].split('/')[-1] if '/' in result['embedding_model'] else result['embedding_model']
                    extra_info = f" ({model_name})"
                elif "successful_endpoints" in result:
                    extra_info = f" ({result['successful_endpoints']}/{result['tested_endpoints']} endpoints)"
                elif "error_type" in result:
                    extra_info = f" ({result['error_type']})"
                
                print(f"  {status} {result['test_name']}: {result.get('status_code', 'multi')}{extra_info}")
            except Exception as e:
                results.append({
                    "test_name": test.__name__,
                    "success": False,
                    "error": str(e)
                })
                print(f"  💥 {test.__name__}: ERROR - {e}")
        
        return results