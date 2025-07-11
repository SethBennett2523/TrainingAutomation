#!/bin/bash
# Quick training script with recall-optimized parameters

echo "🎯 Starting YOLO training optimized for reducing false negatives..."
echo "📊 Key optimizations:"
echo "   • Confidence threshold: 0.05 (aggressive detection)"
echo "   • Image size: 704px (better small object detection)"
echo "   • Enhanced loss weights for recall"
echo "   • Lower anchor threshold for more positive matches"
echo ""

# Check if we have a model directory
if [ ! -d "models" ]; then
    mkdir -p models
    echo "✅ Created models directory"
fi

# Run training with the updated configuration
echo "🚀 Starting training with recall optimization..."
python3 main.py train --config config.yaml

echo ""
echo "✅ Training completed!"
echo "📈 Next steps:"
echo "   1. Check recall metrics in tensorboard logs"
echo "   2. Run threshold tuning: python3 main.py tune-thresholds"
echo "   3. Test inference with lower confidence thresholds"
