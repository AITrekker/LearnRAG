FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./
RUN npm install

# Copy source code
COPY . .

# Make startup script executable
RUN chmod +x startup.sh

EXPOSE 3000

CMD ["./startup.sh"]