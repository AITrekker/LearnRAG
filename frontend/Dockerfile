FROM node:18-alpine

WORKDIR /app

# Copy package files first for better Docker layer caching
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy source code
COPY . .

# Fix line endings and permissions (Mac/Windows compatibility)
RUN apk add --no-cache dos2unix && \
    find . -type f -name "*.sh" -exec dos2unix {} \; && \
    find . -type f -name "*.sh" -exec chmod +x {} \; && \
    chmod 755 public/ && \
    chmod 644 public/index.html

# Verify index.html exists and is readable
RUN ls -la public/index.html && cat public/index.html | head -5

# Create startup script with better error handling
RUN echo '#!/bin/sh' > /app/safe-start.sh && \
    echo 'set -e' >> /app/safe-start.sh && \
    echo 'echo "Starting LearnRAG Frontend..."' >> /app/safe-start.sh && \
    echo 'echo "Working directory: $(pwd)"' >> /app/safe-start.sh && \
    echo 'echo "Node version: $(node --version)"' >> /app/safe-start.sh && \
    echo 'echo "NPM version: $(npm --version)"' >> /app/safe-start.sh && \
    echo 'if [ ! -f public/index.html ]; then' >> /app/safe-start.sh && \
    echo '    echo "ERROR: public/index.html not found!"' >> /app/safe-start.sh && \
    echo '    ls -la public/' >> /app/safe-start.sh && \
    echo '    exit 1' >> /app/safe-start.sh && \
    echo 'fi' >> /app/safe-start.sh && \
    echo 'if [ ! -L /app/public/api_keys.json ]; then' >> /app/safe-start.sh && \
    echo '    echo "Creating symbolic link for api_keys.json..."' >> /app/safe-start.sh && \
    echo '    ln -s /app/output/frontend/public/api_keys.json /app/public/api_keys.json 2>/dev/null || true' >> /app/safe-start.sh && \
    echo 'fi' >> /app/safe-start.sh && \
    echo 'echo "Starting React development server..."' >> /app/safe-start.sh && \
    echo 'exec npm start' >> /app/safe-start.sh && \
    chmod +x /app/safe-start.sh

EXPOSE 3000

# Use the safe startup script
CMD ["/app/safe-start.sh"]