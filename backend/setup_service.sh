#!/bin/bash

# Citi Bike Backend Persistence Setup Script
# This script installs the systemd service to keep the backend running permanently.

echo "🚀 Starting Systemd Service Setup..."

# 1. Check for sudo
if [ "$EUID" -ne 0 ]; then 
  echo "❌ Please run as root (use sudo)"
  exit 1
fi

# 2. Variables
SERVICE_NAME="citibike.service"
SERVICE_PATH="/etc/systemd/system/$SERVICE_NAME"
CURRENT_DIR=$(pwd)
SOURCE_FILE="$CURRENT_DIR/citibike.service"

# 3. Verify source file exists
if [ ! -f "$SOURCE_FILE" ]; then
    echo "❌ Error: $SERVICE_NAME not found in current directory."
    echo "Make sure you are in the 'backend' directory."
    exit 1
fi

echo "📋 Found service file at $SOURCE_FILE"

# 4. Copy to systemd directory
echo "📦 Installing service to $SERVICE_PATH..."
cp "$SOURCE_FILE" "$SERVICE_PATH"

# 5. Reload systemd daemon
echo "🔄 Reloading systemd daemon..."
systemctl daemon-reload

# 6. Enable service on boot
echo "🔗 Enabling service to start on boot..."
systemctl enable $SERVICE_NAME

# 7. Restart service
echo "▶️  Starting service..."
systemctl restart $SERVICE_NAME

# 8. Check status
echo "🔍 Checking service status..."
if systemctl is-active --quiet $SERVICE_NAME; then
    echo "✅ SUCCESS: Service is running!"
    systemctl status $SERVICE_NAME --no-pager | head -n 10
else
    echo "❌ ERROR: Service failed to start."
    systemctl status $SERVICE_NAME --no-pager
    journalctl -u $SERVICE_NAME -n 20 --no-pager
    exit 1
fi
