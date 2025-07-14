#!/usr/bin/env python3
"""
LearnRAG API Test Runner
========================

Quick API tests to verify backend functionality and debug logging issues.
Run this script to test all API endpoints and see request/response patterns.

Usage:
    python run_all_tests.py

Requirements:
    - Backend container running on localhost:8000
    - Frontend container running on localhost:3000 (for API keys)
"""

import sys
import os
import subprocess
import time
import requests

def check_services():
    """Check if required services are running"""
    print("🔍 Checking services...")
    
    services = [
        ("Backend", "http://localhost:8000/health"),
        ("Frontend", "http://localhost:3000/api_keys.json")
    ]
    
    all_running = True
    for service_name, url in services:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✅ {service_name} is running")
            else:
                print(f"❌ {service_name} returned status {response.status_code}")
                all_running = False
        except requests.exceptions.RequestException as e:
            print(f"❌ {service_name} is not accessible: {e}")
            all_running = False
    
    return all_running

def run_tests():
    """Import and run the API tests"""
    try:
        # Add test directory to Python path
        test_dir = os.path.join(os.path.dirname(__file__), 'test')
        sys.path.insert(0, test_dir)
        
        # Import and run tests
        from api_tests import run_api_tests
        return run_api_tests()
        
    except ImportError as e:
        print(f"❌ Error importing tests: {e}")
        return False
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

def main():
    """Main test runner"""
    print("🧪 LearnRAG API Test Runner")
    print("=" * 50)
    
    # Check if services are running
    if not check_services():
        print("\n❌ Some services are not running. Please start the containers:")
        print("   docker-compose up")
        return 1
    
    print("\n" + "=" * 50)
    
    # Run the API tests
    success = run_tests()
    
    if success:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)