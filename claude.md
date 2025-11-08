# MemoryMark - Development Guide for Claude Code

## Project Overview

**MemoryMark** is a GPU memory analysis tool that identifies waste in ML training configurations by simulating full training loops (forward + backward pass) to find optimal batch sizes that maximize GPU utilization.

**Core Value**: "You're wasting 60-70% of your GPU. Here's the proof. Here's the fix. Here's $47k saved."

## Key Differentiator

Unlike naive tools that only test inference (forward pass), MemoryMark simulates complete training loops including **gradient computation** via `loss.backward()`. This ensures recommended batch sizes actually work during real training.

## Tech Stack

- **Backend**: Python 3.10+, Flask, PyTorch 2.0+, HuggingFace Transformers
- **Frontend**: React 18+, Tailwind CSS, hosted on Vercel
- **GPU Compute**: Lambda Labs A10 (24GB VRAM, $0.60/hr)
- **Models**: BERT, GPT-2, ResNet-50

## Task Management with Task Master

This project uses **Claude Task Master** for AI-driven task management. Reference [taskmaster-cli-reference.md](taskmaster-cli-reference.md) for all CLI commands.

### Quick Task Master Setup

```bash
# Initialize project
task-master init

# Parse this PRD to generate tasks
task-master parse-prd --input=prd.txt

# View all tasks
task-master list

# Get next task
task-master next

# View specific task
task-master show task-001

# Update task status
task-master set-status --id=task-001 --status=in-progress
```

### Recommended Workflow

1. **Start**: `task-master parse-prd --input=prd.txt` - Generate task breakdown
2. **Plan**: `task-master list` - Review all tasks
3. **Execute**: `task-master next` - Get AI-recommended next task
4. **Track**: Use `task-master move` to update task statuses
5. **Research**: `task-master research "[query]"` for technical questions

### Token Optimization

```bash
# Use core mode to save tokens (~5K vs ~21K)
export TASK_MASTER_TOOLS=core
```

## Project Structure

```
memorymark/
├── backend/
│   ├── memorymark.py      # Core analysis engine (FULL TRAINING SIMULATION)
│   ├── app.py             # Flask API server
│   ├── requirements.txt   # Python dependencies
│   └── README.md
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── ModelSelector.jsx
│   │   │   ├── AnalyzeButton.jsx
│   │   │   ├── ResultsDisplay.jsx
│   │   │   └── LoadingState.jsx
│   │   ├── api.js         # Backend API calls
│   │   └── index.css
│   ├── package.json
│   └── README.md
├── demo/
│   ├── slides.pdf
│   ├── demo_video.mp4
│   └── demo_script.md
├── prd.txt                # This requirements document
├── taskmaster-cli-reference.md  # Task Master commands
└── claude.md              # This file
```

## Critical Implementation Details

### 1. Full Training Simulation (memorymark.py)

The `test_batch_size()` function MUST include backward pass:

```python
# FORWARD PASS
outputs = model(**inputs)

# CREATE DUMMY LOSS (critical for backward pass)
loss = outputs.logits.mean()

# BACKWARD PASS - THIS IS THE KEY
loss.backward()

# Measure PEAK memory (not current!)
peak_memory_bytes = torch.cuda.max_memory_allocated()
```

### 2. Memory Measurement

- **Use**: `torch.cuda.max_memory_allocated()` - accurate, fast, reliable
- **Avoid**: `nvidia-smi` - low accuracy, slow, flaky
- **Reset**: Call `torch.cuda.reset_peak_memory_stats()` before each test

### 3. Validation Test

Expected ratio for forward+backward vs forward-only: **2-3x**

If ratio ≈ 1x, backward pass isn't running (CRITICAL BUG).

## API Endpoints

### POST /analyze
```json
Request: { "model_name": "bert" }
Response: {
  "status": "success",
  "data": {
    "optimal_batch_size": 40,
    "optimal_memory_gb": 22.1,
    "current_batch_size": 16,
    "waste_gb": 1.9,
    "waste_percent": 63.4,
    "speedup": 2.5,
    "cost_savings_annual": 2847.50
  }
}
```

### GET /health
Returns GPU status and availability.

### GET /models
Returns list of available models (bert, gpt2, resnet).

## Frontend Design

- **Theme**: Dark mode (#0a0a0a background)
- **Accent**: Neon green (#00ff88)
- **Vibe**: Technical, professional "hacker aesthetic"
- **Components**: Model selector, GPU info, analyze button, loading state, results display with memory bars

## Build Timeline (5.5 hours)

| Hour | Task | Critical? |
|------|------|-----------|
| 0-1 | Backend Core - Forward Pass | ✓ |
| 1-1.5 | **Add Backward Pass Logic** | **✓ CRITICAL** |
| 1.5-2.5 | Flask API + Testing | ✓ |
| 2.5-3.5 | Frontend Generation | ✓ |
| 3.5-4.5 | Integration | ✓ |
| 4.5-5.5 | Deployment + Polish | ✓ |

## Deployment

### Backend (Lambda Labs)
```bash
# SSH into instance
ssh ubuntu@<LAMBDA_IP>

# Setup
mkdir memorymark && cd memorymark
python3 -m venv venv
source venv/bin/activate
pip install torch transformers flask flask-cors pillow

# Start server in tmux
tmux new -s memorymark
python3 app.py
# Detach: Ctrl+B, then D
```

### Frontend (Vercel)
1. Push to GitHub
2. Import to Vercel
3. Set env var: `REACT_APP_API_URL=http://<LAMBDA_IP>:5000`
4. Deploy

## Testing Strategy

```bash
# 1. GPU Available
python -c "import torch; print(torch.cuda.is_available())"  # True

# 2. BERT Analysis (with backprop)
python memorymark.py bert  # batch ~32-40 (NOT 56)

# 3. API Health
curl http://localhost:5000/health

# 4. Full Analysis
curl -X POST http://localhost:5000/analyze -H "Content-Type: application/json" -d '{"model_name":"bert"}'
```

## Common Issues

| Issue | Solution |
|-------|----------|
| Results same as forward-only | **CRITICAL**: Backward pass not running. Debug immediately. |
| OOM on small batch_size | Model too large for GPU. Use smaller model. |
| API timeout | Analysis takes 30-60s. Increase frontend timeout to 90s. |
| CORS error | Verify flask-cors installed and `origins="*"` |

## Success Criteria

### MVP
- ✓ Full training simulation (forward + backward) working
- ✓ Analysis works for at least 1 model (BERT)
- ✓ Backend returns accurate batch sizes
- ✓ Frontend displays results visually
- ✓ Live demo works end-to-end

### Winning Features
- ✓ All 3 models working (BERT, GPT-2, ResNet)
- ✓ Technically accurate (backward pass simulation)
- ✓ Professional web UI
- ✓ Cost calculator showing savings
- ✓ Validation test proves backward pass works

## Demo Script (5 minutes)

**0:00-0:30**: Hook - "Who needs bigger GPU?" → "No, you're wasting 60-70%"

**0:30-2:00**: Live analysis on BERT, show red→green bars

**2:00-3:00**: **Technical Flex** - "We simulate FULL training, not just inference. That's why our recommendations actually work when you train. Most tools only test forward pass and give batch sizes that crash during training. Not us."

**3:00-4:00**: Impact - $47k saved in lab, 3x speedup

**4:00-5:00**: Close - Universal problem, technically sound, give URL

## Task Master Integration Notes

When working on this project with Claude Code:

1. **Always start** with `task-master parse-prd --input=prd.txt`
2. **Use** `task-master next` to get AI-recommended task order
3. **Research** technical questions: `task-master research "PyTorch backward pass memory"`
4. **Track progress** with `task-master move` between backlog/in-progress/completed
5. **Set** `TASK_MASTER_TOOLS=core` to minimize token usage

## Cost Breakdown

- Lambda Labs A10: $0.60/hr × 5.5 hrs = $3.30
- Vercel Hosting: Free
- **Total: $3.30**

## Key Takeaway

**Technical accuracy > Feature count**

The backward pass simulation is the killer feature. It takes 30 extra minutes but makes the tool scientifically accurate. This differentiates you from naive implementations and impresses technical judges.

---

**Ready to build?** Start with `task-master init` and `task-master parse-prd --input=prd.txt`
