# Delta-Infinity Deployment Guide

## Prerequisites
- Vercel account
- Lambda Labs account
- GitHub account

## Environment Variables
- `NEXT_PUBLIC_API_URL`: Points to Lambda Labs backend

## Deployment Steps

### 1. Backend Deployment (Lambda Labs)

#### Launch Lambda Labs GPU Instance

1. Go to [lambdalabs.com/service/gpu-cloud](https://lambdalabs.com/service/gpu-cloud)
2. Sign up or sign in
3. Click "Launch Instance"
4. Select: **1x A10 (24GB)** GPU - $0.60/hour
5. Choose region: Any available (prefer us-west or us-east)
6. Select OS: **Ubuntu 22.04 LTS with PyTorch**
7. Configure SSH access (add your SSH key or use password)
8. Click "Launch" and wait 2-3 minutes
9. **Note the public IP address** - you'll need this later

#### Setup Backend on Lambda Labs

SSH into your Lambda Labs instance:
```bash
ssh ubuntu@<LAMBDA_IP_ADDRESS>
```

Install system dependencies:
```bash
sudo apt update
sudo apt install -y python3-pip git tmux
```

Verify GPU availability:
```bash
nvidia-smi  # Should show NVIDIA A10 with 24GB
python3 -c "import torch; print(torch.cuda.is_available())"  # Should print: True
```

Clone or upload backend code:
```bash
# Option 1: If code is on GitHub
git clone <your-repo-url>
cd <project-name>/backend

# Option 2: Upload via SCP from local machine
# From your local machine:
# scp -r backend/ ubuntu@<LAMBDA_IP>:/home/ubuntu/delta-infinity/
```

Setup Python environment:
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Test the analysis engine:
```bash
python memorymark.py bert
# Should output analysis results
```

#### Start Flask Server in tmux

Start a persistent tmux session:
```bash
tmux new -s memorymark
source venv/bin/activate
python app.py
```

Detach from tmux: Press `Ctrl+B`, then `D`

The server will continue running even after you disconnect. To reattach later:
```bash
tmux attach -t memorymark
```

#### Test Backend API

From your local machine, test the API:
```bash
# Health check
curl http://<LAMBDA_IP>:5000/health

# Test analysis (this takes 30-60 seconds)
curl -X POST http://<LAMBDA_IP>:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"model_name": "bert"}'
```

#### Configure Firewall

Ensure Lambda Labs firewall allows port 5000:
- Check Lambda Labs dashboard → Instance → Firewall settings
- Add rule: Allow TCP port 5000 from 0.0.0.0/0 (or specific IPs for security)

### 2. Frontend Deployment (Vercel)

#### Prepare GitHub Repository

1. Initialize git (if not already):
   ```bash
   git init
   git add .
   git commit -m "Initial commit - Delta-Infinity"
   ```

2. Create a GitHub repository and push:
   ```bash
   git remote add origin https://github.com/<username>/<repo-name>.git
   git push -u origin main
   ```

#### Deploy to Vercel

1. Go to [vercel.com](https://vercel.com) and sign in with GitHub
2. Click "New Project"
3. Import your GitHub repository
4. Vercel will auto-detect Next.js
5. Configure project settings:
   - **Framework Preset**: Next.js
   - **Root Directory**: `./` (or leave default)
   - **Build Command**: `npm run build` (default)
   - **Output Directory**: `.next` (default)

#### Configure Environment Variables

In Vercel project settings → Environment Variables:

1. Add variable:
   - **Name**: `NEXT_PUBLIC_API_URL`
   - **Value**: `http://<LAMBDA_IP>:5000`
   - **Environment**: Select all (Production, Preview, Development)

2. Save and redeploy (Vercel will automatically rebuild)

#### Verify Deployment

1. Visit your Vercel deployment URL: `https://<project-name>.vercel.app`
2. Test the complete workflow:
   - Select a model (BERT, GPT-2, or ResNet)
   - Click "Analyze"
   - Wait for results (30-60 seconds)
   - Verify results display correctly

### 3. Connect Frontend to Backend

The frontend automatically connects to the backend using the `NEXT_PUBLIC_API_URL` environment variable.

**Local Development:**
- Set `NEXT_PUBLIC_API_URL=http://localhost:5000` in `.env.local`

**Production:**
- Set `NEXT_PUBLIC_API_URL=http://<LAMBDA_IP>:5000` in Vercel environment variables

### 4. Updating API URL When Lambda IP Changes

Lambda Labs IP addresses change when instances are restarted. To update:

1. **Find new IP address:**
   - Check Lambda Labs dashboard → Your instance → IP address

2. **Update Vercel environment variable:**
   - Go to Vercel project → Settings → Environment Variables
   - Edit `NEXT_PUBLIC_API_URL`
   - Change IP to new address
   - Save (triggers automatic redeploy)

3. **Verify connection:**
   - Test health endpoint: `curl http://<NEW_IP>:5000/health`
   - Test from live site after redeploy completes

## Troubleshooting

### If API doesn't work: Check Lambda Labs instance is running

- Verify Flask is running: `tmux attach -t memorymark`
- Test health endpoint: `curl http://<IP>:5000/health`
- Check Lambda Labs dashboard to ensure instance is active
- Verify GPU is available: `nvidia-smi` (from SSH session)

### If CORS errors: Update backend CORS configuration

- Update `backend/app.py` CORS configuration to allow Vercel domain
- Or use `CORS(app)` to allow all origins (for development)
- Check browser console for specific CORS error messages
- Verify `Access-Control-Allow-Origin` header is present in API responses

### Other Common Issues

**Backend Issues:**
- **Cannot connect to backend from Vercel**: Check Lambda Labs firewall allows port 5000, verify Flask is bound to `0.0.0.0`, not `127.0.0.1`
- **Analysis times out**: Analysis takes 30-60 seconds - this is normal. Check Lambda Labs instance is running and GPU is available

**Frontend Issues:**
- **Build fails on Vercel**: Check Node.js version (should be 18+), verify all dependencies in `package.json`, check build logs in Vercel dashboard
- **API calls fail**: Verify `NEXT_PUBLIC_API_URL` is set correctly in Vercel, check browser console for CORS or network errors, test backend directly with curl
- **Environment variable not working**: Ensure variable name starts with `NEXT_PUBLIC_` for client-side access, redeploy after changing environment variables, clear browser cache

**Lambda Labs Issues:**
- **Instance stopped**: Lambda Labs instances stop after inactivity. Restart from dashboard. Note: IP address may change after restart
- **Cannot SSH into instance**: Check SSH key is added to Lambda Labs account, verify instance is running (not stopped), try password authentication if SSH key fails

## Production Checklist

Before going live, verify:

- [ ] Backend deployed on Lambda Labs and accessible
- [ ] Flask server running in tmux session
- [ ] Health endpoint returns valid response
- [ ] Frontend deployed on Vercel
- [ ] Environment variable `NEXT_PUBLIC_API_URL` set correctly
- [ ] CORS configured to allow Vercel domain
- [ ] All three models (BERT, GPT-2, ResNet) can be analyzed
- [ ] Results display correctly in UI
- [ ] Mobile responsiveness tested
- [ ] Error handling works for backend failures

## Additional Resources

- [Backend Documentation](./backend/README.md)
- [Next.js Documentation](https://nextjs.org/docs)
- [Vercel Deployment Guide](https://vercel.com/docs)
- [Lambda Labs Documentation](https://lambdalabs.com/service/gpu-cloud)

