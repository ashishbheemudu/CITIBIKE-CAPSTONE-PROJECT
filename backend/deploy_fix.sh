#!/bin/bash
# Quick deployment script - pull latest code and restart service

echo "🚀 DEPLOYING EMERGENCY FIX..."

cd /home/ubuntu/CITIBIKE-CAPSTONE-PROJECT/backend

echo "📥 Pulling latest code..."
git pull origin main

echo "🔄 Restarting service..."
sudo systemctl restart citibike

echo "⏳ Waiting for service to start..."
sleep 3

echo "🔍 Checking service status..."
sudo systemctl status citibike --no-pager | head -15

echo "✅ DEPLOYMENT COMPLETE"
echo ""
echo "Test with:"
echo "curl https://3.22.236.184.nip.io/api/stations | head -5"
