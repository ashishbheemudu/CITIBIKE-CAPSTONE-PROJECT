#!/bin/bash

# Stop on error
set -e

echo "🚀 Starting AWS Environment Setup..."

# 1. Update System
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# 2. Install Dependencies
echo "🛠️ Installing Python, Git, Nginx, and system tools..."
sudo apt install -y python3-pip python3-venv git nginx acl

# 3. Clone Repository
# Check if repo exists, if so pull, else clone
if [ -d "CITIBIKE-CAPSTONE-PROJECT" ]; then
    echo "🔄 Repository exists, pulling latest changes..."
    cd CITIBIKE-CAPSTONE-PROJECT
    git pull
else
    echo "⬇️ Cloning repository..."
    git clone https://github.com/ashishbheemudu/CITIBIKE-CAPSTONE-PROJECT.git
    cd CITIBIKE-CAPSTONE-PROJECT
fi

# 4. Setup Backend
echo "🐍 Setting up Python Virtual Environment..."
cd backend
python3 -m venv venv
source venv/bin/activate

echo "📦 Installing Python dependencies (this may take a while)..."
# Upgrade pip first to avoid errors
pip install --upgrade pip
# Install requirements
pip install -r requirements.txt

# 5. Setup Systemd Service
echo "⚙️ Configuring Systemd Service..."
sudo cp citibike.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable citibike
sudo systemctl restart citibike

# 6. Setup Nginx (Proxy Port 80 -> 8000)
echo "🌐 Configuring Nginx..."
# Create a simple Nginx config to proxy to Uvicorn
sudo bash -c 'cat > /etc/nginx/sites-available/citibike <<EOF
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF'

# Enable site and remove default
if [ -f /etc/nginx/sites-enabled/default ]; then
    sudo rm /etc/nginx/sites-enabled/default
fi
if [ ! -f /etc/nginx/sites-enabled/citibike ]; then
    sudo ln -s /etc/nginx/sites-available/citibike /etc/nginx/sites-enabled/
fi

sudo systemctl restart nginx

echo "✅ Setup Complete! The application should be live on port 80."
