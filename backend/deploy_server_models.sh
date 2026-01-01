
#!/bin/bash
set -e

MODELS_DIR="/home/ubuntu/CITIBIKE-CAPSTONE-PROJECT/backend/models"

echo "📦 Backing up old models..."
cp $MODELS_DIR/xgb.json $MODELS_DIR/xgb_backup.json || true
cp $MODELS_DIR/scaler_tree.save $MODELS_DIR/scaler_tree_backup.save || true
cp $MODELS_DIR/scaler_y.save $MODELS_DIR/scaler_y_backup.save || true

echo "🚀 Deploying SERVER-SIDE models..."
if [ -f "$MODELS_DIR/xgb_server.json" ]; then
    cp $MODELS_DIR/xgb_server.json $MODELS_DIR/xgb.json
    echo "✅ Deployed XGBoost Model"
else
    echo "❌ xgb_server.json missing!"
fi

if [ -f "$MODELS_DIR/lgb_server.txt" ]; then
    cp $MODELS_DIR/lgb_server.txt $MODELS_DIR/lgb.txt
    echo "✅ Deployed LightGBM Model"
else
    echo "⚠️ lgb_server.txt missing!"
fi

if [ -f "$MODELS_DIR/cb_server.cbm" ]; then
    cp $MODELS_DIR/cb_server.cbm $MODELS_DIR/cb.cbm
    echo "✅ Deployed CatBoost Model"
else
    echo "⚠️ cb_server.cbm missing!"
fi

if [ -f "$MODELS_DIR/scaler_tree_server.save" ]; then
    cp $MODELS_DIR/scaler_tree_server.save $MODELS_DIR/scaler_tree.save
    echo "✅ Deployed Feature Scaler"
else
    echo "⚠️ scaler_tree_server.save missing!"
fi

if [ -f "$MODELS_DIR/scaler_y_server.save" ]; then
    cp $MODELS_DIR/scaler_y_server.save $MODELS_DIR/scaler_y.save
    echo "✅ Deployed Target Scaler"
else
    echo "⚠️ scaler_y_server.save missing!"
fi

echo "🎉 Model Deployment Complete. Restart service to apply."
