import requests
import json
import time
from typing import Dict, Any

class APITester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.api_key = None
        self.session = requests.Session()
        
    def load_api_keys(self):
        """Load API keys from the frontend public folder"""
        try:
            response = requests.get("http://localhost:3000/api_keys.json")
            if response.status_code == 200:
                data = response.json()
                if data.get("tenants") and len(data["tenants"]) > 0:
                    self.api_key = data["tenants"][0]["api_key"]
                    self.session.headers.update({"X-API-Key": self.api_key})
                    print(f"✅ Loaded API key for tenant: {data['tenants'][0]['name']}")
                    return True
            print("❌ Failed to load API keys")
            return False
        except Exception as e:
            print(f"❌ Error loading API keys: {e}")
            return False
    
    def test_endpoint(self, method: str, endpoint: str, data: Dict = None, expected_status: int = 200) -> Dict[str, Any]:
        """Test a single endpoint and return results"""
        url = f"{self.base_url}{endpoint}"
        start_time = time.time()
        
        try:
            if method.upper() == "GET":
                response = self.session.get(url)
            elif method.upper() == "POST":
                response = self.session.post(url, json=data)
            elif method.upper() == "PUT":
                response = self.session.put(url, json=data)
            elif method.upper() == "DELETE":
                response = self.session.delete(url)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            duration = time.time() - start_time
            
            # Determine status emoji
            if response.status_code == expected_status:
                status_emoji = "✅"
            elif response.status_code < 400:
                status_emoji = "🔄"
            elif response.status_code < 500:
                status_emoji = "❌"
            else:
                status_emoji = "💥"
            
            # Method emoji
            method_emoji = {
                "GET": "🔍",
                "POST": "📝",
                "PUT": "✏️",
                "DELETE": "🗑️"
            }.get(method.upper(), "📡")
            
            print(f"{method_emoji} {method.upper()} {endpoint} {status_emoji} {response.status_code} ({duration:.3f}s)")
            
            result = {
                "method": method.upper(),
                "endpoint": endpoint,
                "status_code": response.status_code,
                "expected_status": expected_status,
                "duration": duration,
                "success": response.status_code == expected_status,
                "response_size": len(response.content) if response.content else 0
            }
            
            # Try to parse JSON response
            try:
                result["response_data"] = response.json()
            except:
                result["response_text"] = response.text[:200] + "..." if len(response.text) > 200 else response.text
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"💥 {method.upper()} {endpoint} ERROR ({duration:.3f}s): {e}")
            return {
                "method": method.upper(),
                "endpoint": endpoint,
                "status_code": None,
                "expected_status": expected_status,
                "duration": duration,
                "success": False,
                "error": str(e)
            }

def run_basic_api_tests():
    """Run basic connectivity API tests"""
    print("🚀 Starting Basic API Tests")
    print("=" * 50)
    
    tester = APITester()
    
    # Load API keys first
    if not tester.load_api_keys():
        print("❌ Cannot run tests without API keys")
        return False
    
    tests = [
        # Basic endpoints
        ("GET", "/", 200),
        ("GET", "/health", 200),
        
        # Auth endpoints
        ("GET", "/api/auth/validate", 200),
        
        # Tenant endpoints
        ("GET", "/api/tenants/info", 200),
        ("GET", "/api/tenants/files", 200),
        ("GET", "/api/tenants/stats", 200),
        
        # Embeddings endpoints
        ("GET", "/api/embeddings/models", 200),
        
        # RAG endpoints
        ("GET", "/api/rag/techniques", 200),
        ("GET", "/api/rag/sessions", 200),
        
        # Test some error cases
        ("GET", "/api/nonexistent", 404),
    ]
    
    results = []
    success_count = 0
    
    for method, endpoint, expected_status in tests:
        result = tester.test_endpoint(method, endpoint, expected_status=expected_status)
        results.append(result)
        if result["success"]:
            success_count += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Basic Test Results: {success_count}/{len(tests)} passed")
    
    # Show failed tests
    failed_tests = [r for r in results if not r["success"]]
    if failed_tests:
        print("\n❌ Failed Tests:")
        for test in failed_tests:
            print(f"  {test['method']} {test['endpoint']} - Expected {test['expected_status']}, Got {test.get('status_code', 'ERROR')}")
            if 'error' in test:
                print(f"    Error: {test['error']}")
    
    return success_count == len(tests)

def run_detailed_api_tests():
    """Run detailed API tests with validation"""
    print("\n🔬 Starting Detailed API Tests")
    print("=" * 50)
    
    # Import test classes
    from test_auth import AuthTests
    from test_tenants import TenantTests
    from test_embeddings import EmbeddingTests
    from test_rag import RagTests
    from test_config import ConfigTests
    
    tester = APITester()
    
    # Load API keys first
    if not tester.load_api_keys():
        print("❌ Cannot run detailed tests without API keys")
        return False
    
    # Initialize test classes
    auth_tests = AuthTests(tester.base_url, tester.session)
    tenant_tests = TenantTests(tester.base_url, tester.session)
    embedding_tests = EmbeddingTests(tester.base_url, tester.session)
    rag_tests = RagTests(tester.base_url, tester.session)
    config_tests = ConfigTests(tester.base_url, tester.session)
    
    all_results = []
    total_success = 0
    total_tests = 0
    
    # Run each test suite
    test_suites = [
        ("🔐 Auth Tests", auth_tests),
        ("🏢 Tenant Tests", tenant_tests), 
        ("🧠 Embedding Tests", embedding_tests),
        ("🔍 RAG Tests", rag_tests),
        ("⚙️ Config Tests", config_tests)
    ]
    
    for suite_name, test_suite in test_suites:
        print(f"\n{suite_name}:")
        suite_results = test_suite.run_all_tests()
        all_results.extend(suite_results)
        
        suite_success = sum(1 for r in suite_results if r["success"])
        total_success += suite_success
        total_tests += len(suite_results)
    
    print("\n" + "=" * 50)
    print(f"📊 Detailed Test Results: {total_success}/{total_tests} passed")
    
    # Show summary of failed tests
    failed_tests = [r for r in all_results if not r["success"]]
    if failed_tests:
        print(f"\n❌ Failed Tests ({len(failed_tests)}):")
        for test in failed_tests:
            error_info = f" - {test.get('error', 'Unknown error')}" if 'error' in test else ""
            print(f"  {test['test_name']}{error_info}")
    
    return total_success == total_tests

def run_api_tests():
    """Run all API tests (basic + detailed)"""
    basic_success = run_basic_api_tests()
    detailed_success = run_detailed_api_tests()
    
    print("\n" + "=" * 50)
    if basic_success and detailed_success:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed!")
        return False

if __name__ == "__main__":
    run_api_tests()