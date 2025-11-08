# MemoryMark Backend

GPU Memory Waste Detection and Optimization Engine

## Overview

MemoryMark analyzes GPU memory usage during ML model training to find optimal batch sizes. Unlike naive tools that only test inference (forward pass), MemoryMark simulates **complete training loops including backward pass** to ensure recommendations actually work in production.

## Key Features

- ✅ **Full Training Simulation**: Tests forward + backward pass for accurate results
- ✅ **Multi-Model Support**: BERT, GPT-2, ResNet-50
- ✅ **REST API**: Easy integration with frontends
- ✅ **CUDA + MPS Support**: Works on NVIDIA GPUs and Apple Silicon
- ✅ **Cost Calculator**: Estimates annual savings

## Technical Stack

- **Python**: 3.10+
- **PyTorch**: 2.9.0+ (CUDA 12.1 or MPS)
- **Transformers**: 4.57.1+ (HuggingFace)
- **Flask**: 3.1.2+
- **Flask-CORS**: 6.0.1+

## Project Structure

```
backend/
├── memorymark.py      # Core GPU memory analysis engine
├── app.py             # Flask REST API server
├── requirements.txt   # Python dependencies
├── venv/              # Python virtual environment
└── README.md          # This file
```

## Installation

### Prerequisites

- Python 3.10 or higher
- NVIDIA GPU with CUDA 12.1+ (for production)
- OR Apple Silicon Mac with MPS (for development)

### Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

### For Lambda Labs (Production - CUDA)

```bash
# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install transformers flask flask-cors pillow
```

### For Mac (Development - MPS)

```bash
# Standard pip install works
pip install -r requirements.txt
```

## Usage

### Command Line (memorymark.py)

Test a single model directly:

```bash
python memorymark.py bert    # Auto-detects device (cuda/mps/cpu)
python memorymark.py gpt2 cuda  # Force CUDA
python memorymark.py resnet mps  # Force Apple Silicon
```

**Output:**
```
============================================================
■ MemoryMark Results - BERT on CUDA
============================================================
Optimal batch size: 40
Current batch size: 16
Waste: 1.9GB (8.0%)
Speedup: 2.5x
Savings: $0.48/run, $48.00/year
============================================================
```

**⚠️ WARNING**: This downloads models (~500MB-2GB) and uses GPU heavily. Only run on Lambda Labs GPU, NOT your local Mac!

### REST API (app.py)

Start the Flask server:

```bash
python app.py
```

Server runs on `http://0.0.0.0:5001` (port 5001 to avoid macOS AirPlay conflict)

#### API Endpoints

##### GET /
API information and documentation

```bash
curl http://localhost:5001/
```

##### GET /health
Health check and GPU status

```bash
curl http://localhost:5001/health
```

**Response:**
```json
{
  "status": "healthy",
  "gpu_available": true,
  "gpu_name": "NVIDIA A10",
  "gpu_memory_total_gb": 24.0,
  "device": "cuda",
  "timestamp": "2025-11-08T12:00:00Z"
}
```

##### GET /models
List available models

```bash
curl http://localhost:5001/models
```

**Response:**
```json
{
  "models": [
    {
      "id": "bert",
      "name": "BERT Base",
      "description": "NLP model - 110M parameters",
      "type": "nlp",
      "huggingface_id": "google-bert/bert-base-uncased"
    },
    ...
  ]
}
```

##### POST /analyze
Run memory analysis (⚠️ GPU-intensive!)

```bash
curl -X POST http://localhost:5001/analyze \
  -H "Content-Type: application/json" \
  -d '{"model_name": "bert"}'
```

**Request:**
```json
{
  "model_name": "bert"  // One of: bert, gpt2, resnet
}
```

**Response (200 OK):**
```json
{
  "status": "success",
  "data": {
    "model_name": "bert",
    "device": "cuda",
    "gpu_total_gb": 24.0,
    "optimal_batch_size": 40,
    "optimal_memory_gb": 22.1,
    "current_batch_size": 16,
    "current_memory_gb": 8.5,
    "waste_gb": 1.9,
    "waste_percent": 8.0,
    "speedup": 2.5,
    "cost_savings_per_run": 0.48,
    "cost_savings_annual": 48.00
  }
}
```

**Response (400 Bad Request):**
```json
{
  "status": "error",
  "error": "Invalid model_name. Must be one of: bert, gpt2, resnet"
}
```

## Testing

### Quick Validation (No GPU Work)

Test that code imports and runs without heavy GPU operations:

```bash
# Syntax check
python -m py_compile memorymark.py app.py

# Import test
python -c "import memorymark; import app; print('✓ All imports work')"

# Device detection
python -c "import memorymark; print(f'Device: {memorymark.get_device()}')"
```

### Endpoint Testing (Lightweight)

```bash
# Start server
python app.py &

# Test endpoints (no model download)
curl http://localhost:5001/
curl http://localhost:5001/health
curl http://localhost:5001/models

# Stop server
pkill -f "python.*app.py"
```

### Full Testing (Lambda Labs Only!)

```bash
# Test BERT analysis (downloads ~500MB, uses GPU heavily)
python memorymark.py bert

# Test via API
curl -X POST http://localhost:5001/analyze \
  -H "Content-Type: application/json" \
  -d '{"model_name": "bert"}'
```

## Deployment

### Lambda Labs GPU

1. **Launch Instance**
   - Go to lambdalabs.com
   - Select: 1x A10 (24GB) - $0.60/hr
   - OS: Ubuntu 22.04 with PyTorch

2. **SSH and Setup**
   ```bash
   ssh ubuntu@<LAMBDA_IP>

   # Clone repo
   git clone https://github.com/yourusername/memorymark.git
   cd memorymark/backend

   # Setup Python
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

   # Test
   python memorymark.py bert
   ```

3. **Run Server in tmux**
   ```bash
   tmux new -s memorymark
   python app.py
   # Detach: Ctrl+B, then D

   # Reattach later
   tmux attach -t memorymark
   ```

4. **Test from Local Machine**
   ```bash
   curl http://<LAMBDA_IP>:5001/health
   ```

## Documentation References

- [PyTorch CUDA Memory Management](https://docs.pytorch.org/docs/stable/generated/torch.cuda.max_memory_allocated.html)
- [PyTorch Reset Peak Memory Stats](https://docs.pytorch.org/docs/stable/generated/torch.cuda.reset_peak_memory_stats.html)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/main/en/model_doc/auto)
- [Flask 3.x Documentation](https://flask.palletsprojects.com/en/stable/quickstart/)
- [Flask-CORS](https://flask-cors.readthedocs.io/)

## Troubleshooting

### Port 5000 Already in Use (macOS)

Port 5000 is often used by AirPlay on macOS. The app is configured to use port 5001 instead.

To disable AirPlay:
```
System Preferences → General → AirDrop & Handoff → Disable "AirPlay Receiver"
```

### CUDA Out of Memory

This is expected - the tool tests batch sizes until OOM to find the limit. The last successful batch size is optimal.

### Models Download Slowly

First run downloads models from HuggingFace (~500MB-2GB). Subsequent runs use cached models from `~/.cache/huggingface/`.

### MPS (Apple Silicon) vs CUDA Results Different

MPS and CUDA have different memory management. For production results, always test on Lambda Labs with CUDA.

## License

MIT

## Contributors

Built for hackathon by [Your Team Name]

## Support

For issues, please contact [your-email@example.com]
