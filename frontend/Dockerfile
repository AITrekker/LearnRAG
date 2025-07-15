FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./
RUN npm install

# Copy source code
COPY . .

# Fix Windows line endings and make startup script executable
RUN apk add --no-cache dos2unix && \
    dos2unix startup.sh && \
    chmod +x startup.sh

# Create a fallback startup script
RUN echo '#!/bin/sh' > /startup-fallback.sh && \
    echo 'if [ ! -L /app/public/api_keys.json ]; then' >> /startup-fallback.sh && \
    echo '    echo "Creating symbolic link for api_keys.json..."' >> /startup-fallback.sh && \
    echo '    ln -s /app/output/frontend/public/api_keys.json /app/public/api_keys.json 2>/dev/null || true' >> /startup-fallback.sh && \
    echo 'fi' >> /startup-fallback.sh && \
    echo 'exec npm start' >> /startup-fallback.sh && \
    chmod +x /startup-fallback.sh

EXPOSE 3000

# Try the main startup script first, fallback to npm start
CMD ["sh", "-c", "if [ -f ./startup.sh ]; then ./startup.sh; else /startup-fallback.sh; fi"]