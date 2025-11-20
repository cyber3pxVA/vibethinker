#!/bin/bash
# Download VibeThinker GGUF models

echo "Downloading VibeThinker-1.5B Q6_K GGUF model (1.46GB)..."
mkdir -p models
cd models

wget -c https://huggingface.co/MaziyarPanahi/VibeThinker-1.5B-GGUF/resolve/main/VibeThinker-1.5B.Q6_K.gguf

echo "Download complete! Model saved to models/VibeThinker-1.5B.Q6_K.gguf"
echo "You can now run: python3 chat_gguf.py"
