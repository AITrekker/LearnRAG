#!/bin/bash
# Native Mac Setup for Apple Silicon GPU Support
# This runs the backend directly on macOS to access the GPU

echo "🍎 Setting up LearnRAG for native macOS with Apple Silicon GPU..."

# Check if we're on Apple Silicon
if [[ $(uname -m) != "arm64" ]]; then
    echo "❌ This script is for Apple Silicon Macs only"
    exit 1
fi

# Check prerequisites
echo "📋 Checking prerequisites..."

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Install with: brew install python"
    exit 1
fi

# Check PostgreSQL
if ! command -v psql &> /dev/null; then
    echo "❌ PostgreSQL not found. Install with: brew install postgresql@16"
    exit 1
fi

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js not found. Install with: brew install node"
    exit 1
fi

echo "✅ Prerequisites checked"

# Create .env file
if [ ! -f .env ]; then
    echo "📝 Creating .env from .env.example..."
    cp .env.example .env
    # Update for native setup
    sed -i '' 's|DATABASE_URL=postgresql://postgres:postgres@postgres:5432/learnrag|DATABASE_URL=postgresql://postgres:postgres@localhost:5432/learnrag|' .env
    sed -i '' 's|REACT_APP_API_URL=http://localhost:8000|REACT_APP_API_URL=http://localhost:8000|' .env
    echo "✅ .env file created for native setup"
fi

# Setup PostgreSQL
echo "🗄️ Setting up PostgreSQL..."
if ! pg_isready -h localhost -p 5432 &> /dev/null; then
    echo "Starting PostgreSQL..."
    brew services start postgresql@16
    sleep 3
fi

# Create database
psql -h localhost -U $(whoami) -d postgres -c "CREATE DATABASE learnrag;" 2>/dev/null || echo "Database might already exist"
psql -h localhost -U $(whoami) -d learnrag -c "CREATE EXTENSION IF NOT EXISTS vector;" 2>/dev/null || echo "Vector extension setup"

# Setup Python environment
echo "🐍 Setting up Python environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate

# Install backend dependencies
echo "📦 Installing backend dependencies..."
cd backend
pip install -r requirements.txt
cd ..

# Install frontend dependencies
echo "📦 Installing frontend dependencies..."
cd frontend
npm install
cd ..

# Create startup scripts
echo "📄 Creating startup scripts..."

cat > start-backend.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting LearnRAG Backend with Apple Silicon GPU support..."
source venv/bin/activate
cd backend
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else False}')
print(f'MPS built: {torch.backends.mps.is_built() if hasattr(torch.backends, \"mps\") else False}')
"
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
EOF

cat > start-frontend.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting LearnRAG Frontend..."
cd frontend
npm start
EOF

chmod +x start-backend.sh start-frontend.sh

echo "🎉 Native macOS setup complete!"
echo ""
echo "To start LearnRAG with Apple Silicon GPU support:"
echo ""
echo "Terminal 1 (Backend with GPU):"
echo "./start-backend.sh"
echo ""
echo "Terminal 2 (Frontend):"
echo "./start-frontend.sh"
echo ""
echo "Then open: http://localhost:3000"
echo ""
echo "🔥 Your Apple Silicon GPU will be automatically detected and used!"
echo "Look for 'Using Apple Silicon GPU (MPS)' in the backend logs."