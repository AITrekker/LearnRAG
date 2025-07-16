#!/bin/bash
# Fix Mac Frontend Issues
echo "🔧 Fixing Mac frontend issues..."

# Ensure index.html exists
if [ ! -f frontend/public/index.html ]; then
    echo "❌ index.html missing, creating..."
    mkdir -p frontend/public
    cat > frontend/public/index.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="theme-color" content="#000000" />
    <meta name="description" content="LearnRAG - Interactive platform for learning RAG techniques" />
    <title>LearnRAG</title>
  </head>
  <body>
    <noscript>You need to enable JavaScript to run this app.</noscript>
    <div id="root"></div>
  </body>
</html>
EOF
    echo "✅ Created index.html"
else
    echo "✅ index.html exists"
fi

# Fix permissions
echo "🔒 Fixing permissions..."
chmod 644 frontend/public/index.html
chmod 755 frontend/public/

# Create required directories
echo "📁 Creating directories..."
mkdir -p data/files cache/models output

echo "🎉 Mac fixes applied!"
echo ""
echo "Now run:"
echo "docker-compose -f docker-compose.mac.yml up --build"