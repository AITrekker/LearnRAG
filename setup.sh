#!/bin/bash
# LearnRAG Setup Script
# Prepares environment for first-time setup

echo "🚀 Setting up LearnRAG..."

# Detect OS
OS="$(uname -s)"
case "${OS}" in
    Linux*)     MACHINE=Linux;;
    Darwin*)    MACHINE=Mac;;
    CYGWIN*)    MACHINE=Cygwin;;
    MINGW*)     MACHINE=MinGw;;
    *)          MACHINE="UNKNOWN:${OS}"
esac

echo "📱 Detected OS: $MACHINE"

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env from .env.example..."
    cp .env.example .env
    echo "✅ .env file created"
else
    echo "✅ .env file already exists"
fi

# Update .env with latest default LLM model
if grep -q "DEFAULT_LLM_MODEL=google/flan-t5-base" .env; then
    echo "🔄 Updating default LLM model to UnifiedQA..."
    if [[ "$MACHINE" == "Mac" ]]; then
        sed -i '' 's/DEFAULT_LLM_MODEL=google\/flan-t5-base/DEFAULT_LLM_MODEL=allenai\/unifiedqa-t5-base/' .env
    else
        sed -i 's/DEFAULT_LLM_MODEL=google\/flan-t5-base/DEFAULT_LLM_MODEL=allenai\/unifiedqa-t5-base/' .env
    fi
    echo "✅ Updated default LLM model"
fi

# Mac-specific fixes
if [[ "$MACHINE" == "Mac" ]]; then
    echo "🍎 Applying Mac-specific fixes..."
    
    # Fix frontend index.html permissions
    if [ -f frontend/public/index.html ]; then
        chmod 644 frontend/public/index.html
        echo "✅ Fixed frontend file permissions"
    fi
    
    # Create directories that might have permission issues
    mkdir -p ./data/files ./cache/models ./output
    echo "✅ Created required directories"
fi

echo "🎓 LearnRAG setup complete!"
echo ""
echo "Next steps:"
if [[ "$MACHINE" == "Mac" ]]; then
    echo "1. Run: docker-compose -f docker-compose.mac.yml up --build"
else
    echo "1. Run: docker-compose up --build"
fi
echo "2. Open: http://localhost:3000"
echo "3. Start learning RAG! 🚀"
echo ""
if [[ "$MACHINE" == "Mac" ]]; then
    echo "🍎 Mac users: Using optimized docker-compose.mac.yml"
    echo "   If you get permission errors, try: sudo chown -R $USER:staff ."
fi