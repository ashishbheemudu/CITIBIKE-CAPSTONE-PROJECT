#!/bin/bash
# Script to copy rf.pkl when download is complete

echo "🔍 Checking rf.pkl download status..."

SOURCE="$HOME/Documents/cap/models_complete/rf.pkl"
DEST="$HOME/Documents/cap/backend/models/rf.pkl"

if [ ! -f "$SOURCE" ]; then
    echo "❌ rf.pkl not found in models_complete/"
    echo "   Expected location: $SOURCE"
    exit 1
fi

# Get file size
SIZE=$(stat -f%z "$SOURCE")
SIZE_GB=$(echo "scale=2; $SIZE/1024/1024/1024" | bc)

echo "📊 Current size: ${SIZE_GB} GB"

# Random Forest should be ~8GB based on your screenshot
if (( $(echo "$SIZE_GB > 7.5" | bc -l) )); then
    echo "✅ File looks complete!"
    echo "Copying to backend/models/..."
    cp "$SOURCE" "$DEST"
    echo "✅ rf.pkl deployed!"
    
    echo ""
    echo "🎉 ALL MODEL FILES DEPLOYED!"
    echo ""
    echo "Next steps:"
    echo "1. Restart dashboard: docker compose down && docker compose up -d --build"
    echo "2. Check logs: docker compose logs backend | grep 'Hydra'"
    echo "3. Open: http://localhost:3000/prediction"
else
    echo "⏳ Still downloading (need ~8GB total)..."
    echo "   Current: ${SIZE_GB} GB"
    echo "   Run this script again when download completes"
fi
