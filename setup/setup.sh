#!/bin/bash
# LearnRAG Setup Script
# Prepares environment for first-time setup

echo "🚀 Setting up LearnRAG..."

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
    sed -i.bak 's/DEFAULT_LLM_MODEL=google\/flan-t5-base/DEFAULT_LLM_MODEL=allenai\/unifiedqa-t5-base/' .env
    echo "✅ Updated default LLM model"
fi

echo "🎓 LearnRAG setup complete!"
echo ""
echo "Next steps:"
echo "1. Run: docker-compose up --build"
echo "2. Open: http://localhost:3000"
echo "3. Start learning RAG! 🚀"