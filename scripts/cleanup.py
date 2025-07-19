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
    if not run_command("docker compose down", "Stopping all containers"):
        print("⚠️  Continuing anyway...")
    
    # Step 2: Remove containers and volumes (most thorough approach)
    if not run_command("docker compose down -v", "Removing containers and volumes"):
        print("⚠️  Continuing anyway...")
    
    # Step 3: Force remove any remaining volumes with project prefix
    project_name = os.path.basename(os.getcwd()).lower()
    
    print(f"🔍 Project name detected: {project_name}")
    
    # Try to remove volumes with different naming patterns
    volume_patterns = [
        f"{project_name}_postgres_data",
        "learnrag_postgres_data", 
        f"{project_name}_models_cache",
        f"{project_name}_huggingface_cache",
        f"{project_name}_transformers_cache"
    ]
    
    for volume in volume_patterns:
        run_command(f"docker volume rm {volume}", f"Removing volume {volume}", ignore_errors=True)
    
    # Step 4: Nuclear option - remove ALL volumes containing project name
    run_command(f"docker volume ls -q | grep {project_name} | xargs -r docker volume rm", 
                f"Removing all {project_name} volumes", ignore_errors=True)
    
    # Step 5: Clean up any dangling volumes
    run_command("docker volume prune -f", "Cleaning up dangling volumes", ignore_errors=True)
    
    # Step 5: Rebuild and start
    print("\n✅ Cleanup complete!")
    print("📋 Next steps:")
    print("   docker compose up -d --build")

if __name__ == "__main__":
    main()