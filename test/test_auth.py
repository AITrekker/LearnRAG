"""
Authentication endpoint tests
"""
import requests
from typing import Dict, Any

class AuthTests:
    def __init__(self, base_url: str, session: requests.Session):
        self.base_url = base_url
        self.session = session
        
    def test_validate_with_valid_key(self) -> Dict[str, Any]:
        """Test API key validation with valid key"""
        response = self.session.get(f"{self.base_url}/api/auth/validate")
        
        result = {
            "test_name": "validate_with_valid_key",
            "status_code": response.status_code,
            "success": response.status_code == 200,
            "endpoint": "/api/auth/validate"
        }
        
        if response.status_code == 200:
            data = response.json()
            result["data"] = data
            result["has_tenant"] = "tenant" in data
            result["valid_response"] = data.get("valid") == True
        
        return result
    
    def test_validate_without_key(self) -> Dict[str, Any]:
        """Test API key validation without key"""
        # Create session without API key
        temp_session = requests.Session()
        response = temp_session.get(f"{self.base_url}/api/auth/validate")
        
        return {
            "test_name": "validate_without_key", 
            "status_code": response.status_code,
            "success": response.status_code == 401,
            "endpoint": "/api/auth/validate",
            "expected_unauthorized": True
        }
    
    def test_validate_with_invalid_key(self) -> Dict[str, Any]:
        """Test API key validation with invalid key"""
        temp_session = requests.Session()
        temp_session.headers.update({"X-API-Key": "invalid-key-123"})
        response = temp_session.get(f"{self.base_url}/api/auth/validate")
        
        return {
            "test_name": "validate_with_invalid_key",
            "status_code": response.status_code, 
            "success": response.status_code == 401,
            "endpoint": "/api/auth/validate",
            "expected_unauthorized": True
        }

    def run_all_tests(self) -> list:
        """Run all auth tests"""
        tests = [
            self.test_validate_with_valid_key,
            self.test_validate_without_key,
            self.test_validate_with_invalid_key
        ]
        
        results = []
        for test in tests:
            try:
                result = test()
                results.append(result)
                status = "✅" if result["success"] else "❌"
                print(f"  {status} {result['test_name']}: {result['status_code']}")
            except Exception as e:
                results.append({
                    "test_name": test.__name__,
                    "success": False,
                    "error": str(e)
                })
                print(f"  💥 {test.__name__}: ERROR - {e}")
        
        return results