#!/bin/sh

# Create symbolic link for api_keys.json if it doesn't exist
if [ ! -L /app/public/api_keys.json ]; then
    echo "Creating symbolic link for api_keys.json..."
    ln -s /app/output/frontend/public/api_keys.json /app/public/api_keys.json
fi

# Start the application
exec npm start