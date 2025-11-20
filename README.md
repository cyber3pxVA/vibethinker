# VibeThinker-1.5B Demo

A demo implementation of VibeThinker-1.5B, a 1.5B parameter reasoning model.

## Features
- 🧮 Excellent at competitive-style math problems (AIME24: 80.3, AIME25: 74.4)
- 💻 Strong code generation (LiveCodeBench v5: 55.9, v6: 51.1)
- ⚡ Only 1.5B parameters but performs like much larger models

## Requirements
- Python 3.8+
- transformers>=4.54.0
- PyTorch with CUDA support (for GPU acceleration)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the demo:
```bash
python vibethinker_demo.py
```

The model will automatically download from Hugging Face on first run (~2GB).

## Recommended Settings
- **Temperature**: 0.6 or 1.0
- **Max tokens**: 40960
- **top_p**: 0.95
- **top_k**: -1

## Best Use Cases
- Competitive-style math problems
- Algorithm coding problems
- Questions in English work best

## License
MIT License

## References
- [Model Card](https://huggingface.co/WeiboAI/VibeThinker-1.5B)
- [GitHub](https://github.com/WeiboAI/VibeThinker)
- [Technical Report](https://huggingface.co/papers/2511.06221)
