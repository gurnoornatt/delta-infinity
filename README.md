# MemoryMark

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0+-ee4c2c.svg)](https://pytorch.org/)
[![Next.js](https://img.shields.io/badge/Next.js-16.0-black)](https://nextjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0-blue)](https://www.typescriptlang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

GPU memory waste detection tool that finds optimal batch sizes for ML training by simulating **full training loops** (forward + backward pass).

## What It Does

Most ML practitioners use default batch sizes (e.g., 16) and waste 60-90% of their GPU memory. MemoryMark finds the maximum batch size your GPU can handle by:

1. Loading your model on GPU
2. Testing incrementing batch sizes (8 → 16 → 24 → 32...)
3. **Simulating full training** with `loss.backward()` to measure gradient memory
4. Measuring peak GPU memory usage for each batch size
5. Finding the optimal batch before OOM (Out of Memory)

**Key Differentiator**: Unlike naive tools that only test inference (forward pass), MemoryMark simulates real training with backward pass. This ensures recommended batch sizes actually work during training.

## Proven Results

Tested on Lambda Labs A10 GPU (24GB VRAM):

| Model | Current Batch | Optimal Batch | GPU Efficiency | Speedup | Annual Savings* |
|-------|---------------|---------------|----------------|---------|-----------------|
| **BERT Base** | 16 | 288 | 96.4% (21.4/22.1 GB) | 18.0x | $113.02 |
| **GPT-2** | 16 | 152 | 96.3% (21.3/22.1 GB) | 9.5x | $107.37 |
| **ResNet-50** | 16 | 264 | 97.1% (21.5/22.1 GB) | 16.5x | $112.73 |

<sup>*Based on 1M training runs/year at Lambda Labs pricing ($0.60/hr for A10)</sup>

## Features

- ✅ **Full Training Simulation** - Forward + backward pass (`loss.backward()`) for accurate memory measurement
- ✅ **Multi-Model Support** - BERT, GPT-2, ResNet-50 (NLP & Vision models)
- ✅ **Live Terminal View** - Real-time GPU analysis visualization with typing animation
- ✅ **REST API** - Flask backend with CORS support
- ✅ **Backward Pass Validation** - CLI tool to prove gradient memory allocation (2-3x overhead)
- ✅ **GPU Auto-Detection** - Supports CUDA, MPS (Apple Silicon), CPU fallback
- ✅ **Cost Calculator** - Annual savings estimates based on GPU usage

## Tech Stack

**Backend**
- Python 3.10+
- PyTorch 2.6.0+ (CUDA/MPS support)
- HuggingFace Transformers 4.50.0+
- Flask 3.0.0+ with flask-cors
- Pillow 10.0.0+

**Frontend**
- Next.js 16.0.0
- React 19.2.0
- TypeScript 5+
- Tailwind CSS 4.1.9
- Radix UI components

**Infrastructure**
- GPU: Lambda Labs A10 (24GB VRAM)
- Backend: Flask on Lambda Labs @ `http://159.54.185.181:5001`
- Frontend: Next.js (Vercel-ready)

## Quick Start

### Backend Setup (Lambda Labs or Local)

```bash
cd backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Flask server
python app.py
```

The server will start on `http://localhost:5001`.

For persistent deployment on Lambda Labs:
```bash
tmux new -s memorymark
python app.py
# Detach: Ctrl+B, then D
```

### Frontend Setup (Local Development)

```bash
# Install dependencies
npm install

# Configure environment
cp .env.example .env.local
# Edit .env.local: NEXT_PUBLIC_API_URL=http://localhost:5001

# Start development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### CLI Usage (Backend Only)

```bash
cd backend
source venv/bin/activate

# Run analysis
python memorymark.py bert
python memorymark.py gpt2
python memorymark.py resnet

# Validate backward pass (proves gradient allocation)
python memorymark.py --validate-backward bert
```

## Project Structure

```
delta-infinity/
├── backend/
│   ├── memorymark.py          # Core GPU analysis engine (761 lines)
│   ├── app.py                 # Flask REST API (269 lines)
│   ├── requirements.txt       # Python dependencies
│   └── README.md              # Backend documentation
├── app/
│   ├── page.tsx              # Main page with analysis flow
│   └── layout.tsx            # Next.js layout
├── components/
│   ├── terminal-view.tsx    # Live terminal simulator
│   ├── results-display.tsx  # Results visualization
│   ├── analyze-button.tsx
│   ├── model-selector.tsx
│   └── ui/                  # Shadcn UI components
├── lib/
│   ├── api.ts              # Backend API client (329 lines)
│   ├── constants.ts        # TypeScript interfaces
│   ├── terminal-simulator.ts # Terminal log generator
│   └── utils.ts
├── .env.example            # Environment template
└── DEPLOYMENT.md           # Production deployment guide
```

## API Reference

### GET `/health`

Check backend status and GPU availability.

**Response:**
```json
{
  "status": "healthy",
  "gpu_available": true,
  "gpu_name": "NVIDIA A10",
  "gpu_memory_total_gb": 22.1,
  "device": "cuda",
  "timestamp": "2025-11-09T03:31:59.352428Z"
}
```

### GET `/models`

List available models.

**Response:**
```json
{
  "models": [
    {
      "id": "bert",
      "name": "BERT Base",
      "description": "Bidirectional Encoder Representations from Transformers",
      "type": "nlp",
      "huggingface_id": "google-bert/bert-base-uncased"
    }
  ]
}
```

### POST `/analyze`

Run GPU memory analysis (30-60 seconds, GPU-intensive).

**Request:**
```json
{
  "model_name": "bert"
}
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "model_name": "bert",
    "device": "cuda",
    "gpu_total_gb": 22.1,
    "optimal_batch_size": 288,
    "optimal_memory_gb": 21.33,
    "current_batch_size": 16,
    "current_memory_gb": 1.62,
    "waste_gb": 0.8,
    "waste_percent": 3.5,
    "speedup": 18.0,
    "cost_savings_annual": 113.33,
    "results": [
      {
        "batch_size": 8,
        "memory_gb": 1.04,
        "success": true
      }
    ]
  }
}
```

## How It Works

### 1. Model Loading
```python
model, tokenizer, model_type, _ = load_model('bert', device='cuda')
```

### 2. Batch Size Testing
```python
for batch_size in range(8, 1000, 8):
    inputs = create_dummy_batch(model_type, batch_size, tokenizer, device)

    # Forward pass
    outputs = model(**inputs)

    # Backward pass - THE CRITICAL STEP
    loss = outputs.logits.mean()
    loss.backward()  # Allocates gradient memory

    # Measure peak memory
    peak_memory = torch.cuda.max_memory_allocated()

    if peak_memory > gpu_total_memory:
        break  # OOM - optimal found
```

### 3. Results Calculation
- **Waste %**: `(gpu_total - optimal_memory) / gpu_total * 100`
- **Speedup**: `optimal_batch / current_batch`
- **Annual Savings**: Based on Lambda Labs pricing and training frequency

## Development

### Testing Locally (CPU/MPS)

```bash
# Backend
cd backend
python memorymark.py bert  # Uses MPS on Mac M-series, CPU otherwise

# Frontend
npm run dev
```

### Type Checking

```bash
npx tsc --noEmit
```

### Backend Validation

```bash
# Verify backward pass is working (ratio should be 2.0-3.0x)
python memorymark.py --validate-backward bert

# Expected output:
# Ratio: 2.66x
# ✓ PASS: Ratio is within expected range [2.0-3.0x]
```

## Deployment

See [DEPLOYMENT.md](./DEPLOYMENT.md) for production deployment guide including:
- Lambda Labs GPU setup
- Flask server configuration
- Firewall rules
- tmux session management
- Vercel frontend deployment

**Current Production Backend**: `http://159.54.185.181:5001`

## Why This Matters

**The Problem**: Default batch sizes waste GPU memory. A researcher using batch size 16 on a 24GB GPU might only use 1.6GB (7% utilization), wasting 22.4GB (93%).

**The Impact**:
- 10-18x slower training
- Higher GPU costs
- Longer time to results

**The Solution**: MemoryMark finds the optimal batch size in 30-60 seconds, providing:
- Faster training (10-18x speedup)
- Lower costs ($100-113/year savings per model)
- Better GPU utilization (90%+ efficiency)

## Key Technical Decisions

1. **Backward pass is mandatory** - Gradient memory is 2-3x larger than forward-only. Tools that skip this recommend batch sizes that OOM during real training.
2. **torch.compile NOT used** - JIT compilation overhead (~9.5s) outweighs benefits for one-time batch testing.
3. **Port 5001** - Avoids macOS AirPlay conflict on port 5000.
4. **90s timeout** - Analysis takes 30-60s; frontend needs buffer for slow networks.

## License

MIT License - see [LICENSE](./LICENSE) file for details.

## Contributing

Contributions welcome! This was built for a hackathon but can be extended to support:
- More models (LLaMA, Mistral, ViT, etc.)
- Mixed precision (FP16, BF16)
- Multi-GPU analysis
- Gradient accumulation recommendations
- Custom dataset support

---

**Built with** ⚡ real GPU testing on Lambda Labs A10
