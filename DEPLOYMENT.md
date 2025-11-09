# MemoryMark - Production Deployment Guide

**Last Updated**: 2025-11-08
**Production URL**: http://159.54.185.181:5001

---

## Production Environment

### Lambda Labs GPU Instance
- **GPU**: NVIDIA A10 (24GB VRAM)
- **Cost**: $0.60/hour
- **IP Address**: 159.54.185.181
- **OS**: Ubuntu 22.04 with PyTorch
- **Python**: 3.10+
- **Backend Location**: `~/delta-infinity/backend`

### Access
```bash
ssh ubuntu@159.54.185.181
```

---

## Backend Deployment (Flask API)

### 1. Initial Setup (Already Complete)

The backend is already deployed and configured. Location:
```bash
cd ~/delta-infinity/backend
```

### 2. Flask Server Management

#### Start Flask Server
```bash
# Navigate to backend
cd ~/delta-infinity/backend

# Activate virtual environment
source venv/bin/activate

# Start Flask in tmux (persistent session)
tmux new -s memorymark
python app.py

# Detach from tmux: Ctrl+B, then D
```

#### View Flask Logs
```bash
# Attach to running tmux session
tmux attach -t memorymark

# Detach without stopping: Ctrl+B, then D
```

#### Restart Flask Server
```bash
# Attach to tmux session
tmux attach -t memorymark

# Stop Flask: Ctrl+C

# Pull latest code
cd ~/delta-infinity
git pull

# Restart Flask
cd backend
python app.py

# Detach: Ctrl+B, then D
```

#### Stop Flask Server
```bash
# Attach and stop
tmux attach -t memorymark
# Press Ctrl+C

# Or kill the session entirely
tmux kill-session -t memorymark
```

---

## API Endpoints

### Base URL
```
http://159.54.185.181:5001
```

### Endpoints

#### 1. Health Check
**GET** `/health`

Returns API status and GPU information.

```bash
curl http://159.54.185.181:5001/health
```

**Response**:
```json
{
  "status": "healthy",
  "gpu_available": true,
  "gpu_name": "NVIDIA A10",
  "gpu_memory_total_gb": 22.1,
  "device": "cuda",
  "timestamp": "2025-11-08T12:00:00Z"
}
```

#### 2. List Models
**GET** `/models`

Returns available models for analysis.

```bash
curl http://159.54.185.181:5001/models
```

**Response**:
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
    {
      "id": "gpt2",
      "name": "GPT-2",
      "description": "NLP model - 124M parameters",
      "type": "nlp",
      "huggingface_id": "openai-community/gpt2"
    },
    {
      "id": "resnet",
      "name": "ResNet-50",
      "description": "Vision model - 26M parameters",
      "type": "vision",
      "huggingface_id": "microsoft/resnet-50"
    }
  ]
}
```

#### 3. Analyze Model (GPU-Intensive)
**POST** `/analyze`

Runs full GPU memory analysis with backward pass simulation.

**⚠️ WARNING**: This operation:
- Takes 15-45 seconds depending on model
- Uses GPU heavily
- Downloads models on first run (~500MB-2GB)

```bash
curl -X POST http://159.54.185.181:5001/analyze \
  -H "Content-Type: application/json" \
  -d '{"model_name": "bert"}' \
  --max-time 120
```

**Request Body**:
```json
{
  "model_name": "bert"  // One of: bert, gpt2, resnet
}
```

**Response (Success)**:
```json
{
  "status": "success",
  "data": {
    "model_name": "bert",
    "device": "cuda",
    "gpu_total_gb": 22.1,
    "optimal_batch_size": 288,
    "optimal_memory_gb": 21.4,
    "current_batch_size": 16,
    "current_memory_gb": 1.2,
    "waste_gb": 0.8,
    "waste_percent": 3.6,
    "speedup": 18.0,
    "cost_savings_per_run": 0.42,
    "cost_savings_annual": 113.02,
    "results": [...]
  }
}
```

**Response (Error)**:
```json
{
  "status": "error",
  "error": "Invalid model_name. Must be one of: bert, gpt2, resnet"
}
```

---

## Verified Test Results

All 3 models tested on Lambda Labs A10 GPU (CUDA):

### BERT
- **Optimal Batch Size**: 288 (up from 16)
- **GPU Utilization**: 96.4% (21.4GB / 22.1GB)
- **Speedup**: 18.0x faster
- **Waste**: 3.6% (0.8GB unused)
- **Annual Savings**: $113.02

### GPT-2
- **Optimal Batch Size**: 152 (up from 16)
- **GPU Utilization**: 96.3% (21.29GB / 22.1GB)
- **Speedup**: 9.5x faster
- **Waste**: 3.7% (0.8GB unused)
- **Annual Savings**: $107.37

### ResNet-50
- **Optimal Batch Size**: 264 (up from 16)
- **GPU Utilization**: 97.1% (21.45GB / 22.1GB)
- **Speedup**: 16.5x faster
- **Waste**: 2.9% (0.7GB unused)
- **Annual Savings**: $112.73

---

## Firewall Configuration

**CRITICAL**: Port 5001 must be open for external access.

### Lambda Labs Dashboard
1. Go to: https://cloud.lambdalabs.com/instances
2. Click on your instance
3. Navigate to "Firewall" tab
4. Add rule:
   - **Type**: TCP
   - **Port**: 5001
   - **Source**: 0.0.0.0/0
   - **Description**: Flask API Server

---

## Frontend Integration

### Environment Variables

Create `.env.local` in frontend:
```bash
NEXT_PUBLIC_API_URL=http://159.54.185.181:5001
```

### API Client Usage

```typescript
import { analyzeModel, getHealth, getModels } from '@/lib/api';

// Check backend health
const health = await getHealth();

// Get available models
const models = await getModels();

// Run analysis
const result = await analyzeModel('bert');
console.log(`Optimal batch: ${result.optimalBatchSize}`);
```

---

## Troubleshooting

### Issue: API Not Responding

**Check Flask is Running**:
```bash
tmux attach -t memorymark
# You should see Flask output. If not, restart Flask.
```

**Check Firewall**:
```bash
# From Lambda Labs instance
curl http://localhost:5001/health

# If this works but external doesn't, check firewall rule
```

### Issue: Model Analysis Fails

**Check GPU Availability**:
```bash
nvidia-smi
# Should show A10 GPU with 22GB memory
```

**Check Disk Space**:
```bash
df -h
# Models cache in ~/.cache/huggingface/ (~2GB per model)
```

### Issue: Connection Timeout

- Analysis takes 15-45 seconds, ensure timeout is set to 120s
- Check Lambda Labs instance is running (not stopped/terminated)

### Issue: OOM (Out of Memory)

This is **expected behavior** - the tool tests batch sizes until OOM to find the limit. The last successful batch size before OOM is optimal.

---

## Updating Code

```bash
# On Lambda Labs instance
cd ~/delta-infinity
git pull

# Restart Flask
tmux attach -t memorymark
# Ctrl+C to stop
python backend/app.py
# Ctrl+B, D to detach
```

---

## Cost Management

### Hourly Cost
- **A10 GPU**: $0.60/hour
- **Monthly (24/7)**: ~$432
- **Recommended**: Stop instance when not in use

### Stop Instance
```bash
# From Lambda Labs dashboard
# Click instance → Stop

# Restart when needed:
# Click instance → Start
# Note: IP address may change after restart
```

---

## Security Notes

- **No Authentication**: API is currently open (demo mode)
- **For Production**: Add API keys, rate limiting, HTTPS
- **Firewall**: Currently allows all IPs (0.0.0.0/0)
- **Recommended**: Restrict to specific IPs in production

---

## Monitoring

### Check API Status
```bash
# Health check
curl http://159.54.185.181:5001/health

# Should return:
# {"status": "healthy", "gpu_available": true, ...}
```

### Check GPU Usage
```bash
# SSH into Lambda Labs
ssh ubuntu@159.54.185.181

# Watch GPU usage in real-time
watch -n 1 nvidia-smi
```

---

## Quick Reference

| Action | Command |
|--------|---------|
| SSH to server | `ssh ubuntu@159.54.185.181` |
| View Flask logs | `tmux attach -t memorymark` |
| Restart Flask | `tmux attach -t memorymark` → Ctrl+C → `python app.py` |
| Health check | `curl http://159.54.185.181:5001/health` |
| Test BERT | `curl -X POST http://159.54.185.181:5001/analyze -H "Content-Type: application/json" -d '{"model_name":"bert"}'` |
| Check GPU | `nvidia-smi` |
| Pull latest code | `cd ~/delta-infinity && git pull` |

---

## Support

For issues or questions:
- Check tmux logs: `tmux attach -t memorymark`
- Review error messages in Flask output
- Verify GPU with `nvidia-smi`
- Check firewall rules in Lambda Labs dashboard

---

**Production URL**: http://159.54.185.181:5001
**Status**: ✅ DEPLOYED AND TESTED (All 3 models working)
**Last Test**: 2025-11-08
