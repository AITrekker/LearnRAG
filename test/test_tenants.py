"""
Tenant endpoint tests
"""
import requests
from typing import Dict, Any

class TenantTests:
    def __init__(self, base_url: str, session: requests.Session):
        self.base_url = base_url
        self.session = session
        
    def test_get_tenant_info(self) -> Dict[str, Any]:
        """Test getting current tenant information"""
        response = self.session.get(f"{self.base_url}/api/tenants/info")
        
        result = {
            "test_name": "get_tenant_info",
            "status_code": response.status_code,
            "success": response.status_code == 200,
            "endpoint": "/api/tenants/info"
        }
        
        if response.status_code == 200:
            data = response.json()
            result["data"] = data
            
            # Validate response structure
            required_fields = ["id", "slug", "name", "file_count", "embedding_count", "created_at"]
            result["has_required_fields"] = all(field in data for field in required_fields)
            result["tenant_name"] = data.get("name")
            result["file_count"] = data.get("file_count", 0)
            result["embedding_count"] = data.get("embedding_count", 0)
        
        return result
    
    def test_get_tenant_files(self) -> Dict[str, Any]:
        """Test getting tenant files list"""
        response = self.session.get(f"{self.base_url}/api/tenants/files")
        
        result = {
            "test_name": "get_tenant_files",
            "status_code": response.status_code,
            "success": response.status_code == 200,
            "endpoint": "/api/tenants/files"
        }
        
        if response.status_code == 200:
            data = response.json()
            result["data"] = data
            result["is_list"] = isinstance(data, list)
            result["file_count"] = len(data) if isinstance(data, list) else 0
            
            # Check first file structure if any files exist
            if isinstance(data, list) and len(data) > 0:
                first_file = data[0]
                file_fields = ["id", "filename", "file_type", "file_size", "created_at"]
                result["first_file_valid"] = all(field in first_file for field in file_fields)
        
        return result
    
    def test_get_tenant_stats(self) -> Dict[str, Any]:
        """Test getting tenant statistics"""
        response = self.session.get(f"{self.base_url}/api/tenants/stats")
        
        result = {
            "test_name": "get_tenant_stats",
            "status_code": response.status_code,
            "success": response.status_code == 200,
            "endpoint": "/api/tenants/stats"
        }
        
        if response.status_code == 200:
            data = response.json()
            result["data"] = data
            
            # Validate stats structure
            expected_fields = ["total_files", "total_embeddings", "file_types", "embedding_models"]
            result["has_expected_fields"] = all(field in data for field in expected_fields)
        
        return result
    
    def test_sync_files_get_status(self) -> Dict[str, Any]:
        """Test getting file sync status (should be GET, not POST for status)"""
        # This tests if the sync endpoint exists and handles GET properly
        response = self.session.get(f"{self.base_url}/api/tenants/sync-files")
        
        result = {
            "test_name": "sync_files_get_status",
            "status_code": response.status_code,
            "success": response.status_code in [200, 405],  # 405 = Method Not Allowed is OK
            "endpoint": "/api/tenants/sync-files"
        }
        
        return result

    def run_all_tests(self) -> list:
        """Run all tenant tests"""
        tests = [
            self.test_get_tenant_info,
            self.test_get_tenant_files,
            self.test_get_tenant_stats,
            self.test_sync_files_get_status
        ]
        
        results = []
        for test in tests:
            try:
                result = test()
                results.append(result)
                status = "✅" if result["success"] else "❌"
                extra_info = ""
                
                # Add useful info for display
                if "file_count" in result:
                    extra_info = f" ({result['file_count']} files)"
                elif "tenant_name" in result:
                    extra_info = f" ({result['tenant_name']})"
                
                print(f"  {status} {result['test_name']}: {result['status_code']}{extra_info}")
            except Exception as e:
                results.append({
                    "test_name": test.__name__,
                    "success": False,
                    "error": str(e)
                })
                print(f"  💥 {test.__name__}: ERROR - {e}")
        
        return results