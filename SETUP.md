# LearnRAG Setup Guide

## Quick Start 

### Mac Users (Recommended)
```bash
# Clone and setup
git clone <repository-url>
cd LearnRAG

# Automated setup with Mac fixes
./setup.sh

# Start with Mac-optimized compose file
docker-compose -f docker-compose.mac.yml up --build
```

### Linux/Windows Users
```bash
# Clone and setup  
git clone <repository-url>
cd LearnRAG

# Automated setup
./setup.sh

# Start normally
docker-compose up --build
```

### Manual Setup (Any OS)
```bash
# Copy environment template
cp .env.example .env

# Mac: Use Mac compose file
docker-compose -f docker-compose.mac.yml up --build

# Linux/Windows: Use default
docker-compose up --build
```

## Environment File Purpose

### `.env.example`
- **Template file** with all available configuration options
- **Safe to commit** to git (no secrets)
- **Starting point** for your custom configuration

### `.env` 
- **Your actual configuration** used by Docker Compose
- **NOT committed** to git (in .gitignore)
- **Created from** .env.example

## Default Models (Optimized for Literature)

- **Embedding**: `sentence-transformers/all-MiniLM-L6-v2` (fast, good quality)
- **LLM**: `allenai/unifiedqa-t5-base` (optimized for Q&A)

## Common Issues

### Mac: "Could not find index.html" Error
**Solution**: Use Mac-specific setup:
```bash
./setup.sh  # Applies Mac fixes automatically  
docker-compose -f docker-compose.mac.yml up --build
```

**Alternative if above fails**:
```bash
# Manual fix - create missing file
mkdir -p frontend/public
cp frontend/public/index.html.backup frontend/public/index.html 2>/dev/null || \
echo '<!DOCTYPE html><html><head><title>LearnRAG</title></head><body><div id="root"></div></body></html>' > frontend/public/index.html

# Then try again
docker-compose -f docker-compose.mac.yml up --build
```

### Mac: Permission Errors
**Solution**: Fix ownership and try again:
```bash
sudo chown -R $USER:staff .
./setup.sh
docker-compose -f docker-compose.mac.yml up --build
```

### "Missing .env file" Error  
**Solution**: Run `cp .env.example .env` or use the setup script

### "SentencePiece not found" Error
**Solution**: Rebuild with `docker-compose up --build` (includes sentencepiece now)

### Mac Hot Reload Not Working
The Mac compose file includes `WATCHPACK_POLLING=true` to fix React hot reload

### Fresh Install Steps
1. `./setup.sh` (auto-detects OS and applies fixes)
2. **Mac**: `docker-compose -f docker-compose.mac.yml up --build`  
3. **Linux/Windows**: `docker-compose up --build`
4. Open http://localhost:3000
5. Start learning RAG! 🎓

## Customization

Edit `.env` to change:
- Model selections
- Chunk sizes  
- API timeouts
- Cache directories
- Database settings

Never edit `.env.example` directly - it's the template for others.