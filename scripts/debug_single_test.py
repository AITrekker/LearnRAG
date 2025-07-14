#!/usr/bin/env python3
"""
Debug individual API tests
"""
import sys
import requests

def test_auth_without_key():
    """Test the failing auth validation without key"""
    print("🔍 Testing auth validation without API key...")
    
    # Create session without API key
    temp_session = requests.Session()
    response = temp_session.get("http://localhost:8000/api/auth/validate")
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
    
    return response

def test_embedding_status():
    """Test the failing embedding status endpoint"""
    print("🔍 Testing embedding status endpoint...")
    
    # Load API key
    api_keys_response = requests.get("http://localhost:3000/api_keys.json")
    if api_keys_response.status_code == 200:
        data = api_keys_response.json()
        api_key = data["tenants"][0]["api_key"]
        
        session = requests.Session()
        session.headers.update({"X-API-Key": api_key})
        
        response = session.get("http://localhost:8000/api/embeddings/status")
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        return response
    else:
        print("❌ Could not load API keys")
        return None

def test_rag_techniques():
    """Test the failing RAG techniques endpoint"""
    print("🔍 Testing RAG techniques endpoint...")
    
    # Load API key
    api_keys_response = requests.get("http://localhost:3000/api_keys.json")
    if api_keys_response.status_code == 200:
        data = api_keys_response.json()
        api_key = data["tenants"][0]["api_key"]
        
        session = requests.Session()
        session.headers.update({"X-API-Key": api_key})
        
        response = session.get("http://localhost:8000/api/rag/techniques")
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        return response
    else:
        print("❌ Could not load API keys")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 debug_single_test.py <test_name>")
        print("Available tests: auth_without_key, embedding_status, rag_techniques")
        sys.exit(1)
    
    test_name = sys.argv[1]
    
    if test_name == "auth_without_key":
        test_auth_without_key()
    elif test_name == "embedding_status":
        test_embedding_status()
    elif test_name == "rag_techniques":
        test_rag_techniques()
    else:
        print(f"Unknown test: {test_name}")