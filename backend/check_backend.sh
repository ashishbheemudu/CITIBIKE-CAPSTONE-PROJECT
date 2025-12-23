#!/bin/bash
# Backend diagnostic script - Run this on EC2

echo "=== CITIBIKE BACKEND DIAGNOSTIC ==="

echo -e "\n1. SERVICE STATUS:"
sudo systemctl status citibike.service --no-pager | head -20

echo -e "\n2. RECENT LOGS (Last 50 lines):"
sudo journalctl -u citibike.service -n 50 --no-pager

echo -e "\n3. PROCESS CHECK:"
ps aux | grep uvicorn | grep -v grep

echo -e "\n4. PORT CHECK:"
sudo netstat -tlnp | grep 8000

echo -e "\n5. DISK SPACE:"
df -h /home/ubuntu

echo -e "\n6. MEMORY:"
free -h

echo -e "\n7. DATA FILES:"
ls -lh /home/ubuntu/CITIBIKE-CAPSTONE-PROJECT/backend/data/v1_core/*.parquet 2>/dev/null | head -5

echo -e "\n=== END DIAGNOSTIC ==="
