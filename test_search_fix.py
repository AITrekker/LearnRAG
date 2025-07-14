#!/usr/bin/env python3
"""
Test if RAG search works after fixing chunk_metadata column
"""
import requests

def test_search_fix():
    """Test RAG search with minimal valid data after schema fix"""
    print("🔍 Testing RAG search after database schema fix...")
    
    # Load API key
    api_keys_response = requests.get("http://localhost:3000/api_keys.json")
    if api_keys_response.status_code == 200:
        data = api_keys_response.json()
        api_key = data["tenants"][0]["api_key"]
        
        session = requests.Session()
        session.headers.update({"X-API-Key": api_key})
        
        # Test with minimal valid payload
        minimal_payload = {
            "query": "test query",
            "embedding_model": "all-MiniLM-L6-v2",
            "rag_technique": "basic_similarity",
            "top_k": 5
        }
        response = session.post("http://localhost:8000/api/rag/search", json=minimal_payload)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        return response
    else:
        print("❌ Could not load API keys")
        return None

if __name__ == "__main__":
    test_search_fix()