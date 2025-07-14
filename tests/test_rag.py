"""
RAG endpoint tests
"""
import requests
from typing import Dict, Any

class RagTests:
    def __init__(self, base_url: str, session: requests.Session):
        self.base_url = base_url
        self.session = session
        
    def test_get_rag_techniques(self) -> Dict[str, Any]:
        """Test getting available RAG techniques"""
        response = self.session.get(f"{self.base_url}/api/rag/techniques")
        
        result = {
            "test_name": "get_rag_techniques",
            "status_code": response.status_code,
            "success": response.status_code == 200,
            "endpoint": "/api/rag/techniques"
        }
        
        if response.status_code == 200:
            data = response.json()
            result["data"] = data
            result["is_list"] = isinstance(data, list)
            result["technique_count"] = len(data) if isinstance(data, list) else 0
            
            # Check if techniques have expected structure
            if isinstance(data, list) and len(data) > 0:
                first_technique = data[0]
                if isinstance(first_technique, dict):
                    result["first_technique_has_name"] = "name" in first_technique
                    result["first_technique_name"] = first_technique.get("name")
        
        return result
    
    def test_get_rag_sessions(self) -> Dict[str, Any]:
        """Test getting RAG search sessions"""
        response = self.session.get(f"{self.base_url}/api/rag/sessions")
        
        result = {
            "test_name": "get_rag_sessions",
            "status_code": response.status_code,
            "success": response.status_code == 200,
            "endpoint": "/api/rag/sessions"
        }
        
        if response.status_code == 200:
            data = response.json()
            result["data"] = data
            result["is_list"] = isinstance(data, list)
            result["session_count"] = len(data) if isinstance(data, list) else 0
            
            # Check session structure if any exist
            if isinstance(data, list) and len(data) > 0:
                first_session = data[0]
                session_fields = ["id", "query", "rag_technique", "created_at"]
                result["first_session_valid"] = all(field in first_session for field in session_fields)
        
        return result
    
    def test_search_validation(self) -> Dict[str, Any]:
        """Test RAG search with invalid data (validation test)"""
        # Test with empty/invalid payload
        invalid_payload = {}
        response = self.session.post(f"{self.base_url}/api/rag/search", json=invalid_payload)
        
        result = {
            "test_name": "search_validation",
            "status_code": response.status_code,
            "success": response.status_code in [400, 422],  # Should fail validation
            "endpoint": "/api/rag/search",
            "expected_validation_error": True
        }
        
        return result
    
    def test_search_with_minimal_data(self) -> Dict[str, Any]:
        """Test RAG search with minimal valid data"""
        # Test with minimal valid payload
        minimal_payload = {
            "query": "test query",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "rag_technique": "similarity_search",
            "top_k": 5
        }
        response = self.session.post(f"{self.base_url}/api/rag/search", json=minimal_payload)
        
        result = {
            "test_name": "search_with_minimal_data",
            "status_code": response.status_code,
            "success": response.status_code in [200, 400],  # 400 OK if no embeddings exist
            "endpoint": "/api/rag/search",
            "method": "POST"
        }
        
        if response.status_code == 200:
            data = response.json()
            result["data"] = data
            # Validate response structure
            expected_fields = ["query", "results", "embedding_model", "rag_technique", "total_results"]
            result["has_required_fields"] = all(field in data for field in expected_fields)
            result["result_count"] = data.get("total_results", 0)
            result["success"] = result["success"] and result["has_required_fields"]
        elif response.status_code == 400:
            # Likely no embeddings available for search
            result["likely_no_embeddings"] = True
        
        return result
    
    def test_get_compare_placeholder(self) -> Dict[str, Any]:
        """Test RAG comparison endpoint (placeholder implementation)"""
        compare_payload = {
            "queries": ["test query 1", "test query 2"],
            "techniques": ["similarity_search"]
        }
        response = self.session.post(f"{self.base_url}/api/rag/compare", json=compare_payload)
        
        result = {
            "test_name": "get_compare_placeholder",
            "status_code": response.status_code,
            "success": response.status_code == 200,
            "endpoint": "/api/rag/compare",
            "method": "POST",
            "is_placeholder": True
        }
        
        if response.status_code == 200:
            data = response.json()
            result["data"] = data
            # Should return placeholder message
            result["is_placeholder_response"] = "coming in Phase 3" in str(data).lower()
        
        return result

    def run_all_tests(self) -> list:
        """Run all RAG tests"""
        tests = [
            self.test_get_rag_techniques,
            self.test_get_rag_sessions,
            self.test_search_validation,
            self.test_search_with_minimal_data,
            self.test_get_compare_placeholder
        ]
        
        results = []
        for test in tests:
            try:
                result = test()
                results.append(result)
                status = "✅" if result["success"] else "❌"
                extra_info = ""
                
                # Add useful info for display
                if "technique_count" in result:
                    extra_info = f" ({result['technique_count']} techniques)"
                elif "session_count" in result:
                    extra_info = f" ({result['session_count']} sessions)"
                elif "result_count" in result:
                    extra_info = f" ({result['result_count']} results)"
                elif result.get("likely_no_embeddings"):
                    extra_info = " (no embeddings available)"
                elif result.get("is_placeholder"):
                    extra_info = " (placeholder)"
                
                print(f"  {status} {result['test_name']}: {result['status_code']}{extra_info}")
            except Exception as e:
                results.append({
                    "test_name": test.__name__,
                    "success": False,
                    "error": str(e)
                })
                print(f"  💥 {test.__name__}: ERROR - {e}")
        
        return results