#!/usr/bin/env python3
"""
LearnRAG Database Cleanup Script
Completely removes database and starts fresh
"""
import subprocess
import sys
import os

def run_command(cmd, description, ignore_errors=False):
    """Run a command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0 and not ignore_errors:
            print(f"❌ Error: {result.stderr}")
            return False
        else:
            if result.stdout:
                print(f"   {result.stdout.strip()}")
            return True
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def main():
    """Main cleanup process"""
    print("🧹 LearnRAG Database Cleanup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('docker-compose.yml'):
        print("❌ Error: docker-compose.yml not found. Run this script from the LearnRAG root directory.")
        sys.exit(1)
    
    # Step 1: Stop all containers
    if not run_command("docker-compose down", "Stopping all containers"):
        print("⚠️  Continuing anyway...")
    
    # Step 2: Remove containers and volumes
    if not run_command("docker-compose down --volumes", "Removing containers and volumes"):
        print("⚠️  Continuing anyway...")
    
    # Step 3: Try to remove the specific postgres volume (ignore errors)
    run_command("docker volume rm learnrag_postgres_data", "Removing postgres volume", ignore_errors=True)
    
    # Step 4: Clean up any dangling volumes
    run_command("docker volume prune -f", "Cleaning up dangling volumes", ignore_errors=True)
    
    # Step 5: Rebuild and start
    print("\n✅ Cleanup complete!")
    print("📋 Next steps:")
    print("   docker compose up -d --build")

if __name__ == "__main__":
    main()