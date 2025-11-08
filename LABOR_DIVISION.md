# MemoryMark - Labor Division Plan

## Overview
This document divides the 50 subtasks between two team members to ensure optimal task alignment and project success.

---

## 🔴 YOU - 35 Critical Tasks

### Task 1: Backend Setup (All 4 subtasks)
- ✅ **1.1** Create backend directory structure
- ✅ **1.2** Setup requirements.txt with modern dependencies
- ✅ **1.3** Create initial README
- ✅ **1.4** Setup git and Python environment

**Why YOU:** Foundation setup requires understanding of Python environments, dependency management, and project structure.

---

### Task 2: Core Memory Analysis Engine (All 6 subtasks)
- ✅ **2.1** Implement load_model() with torch.compile
- ✅ **2.2** Implement create_dummy_batch()
- ✅ **2.3** Implement test_batch_size() with backward pass
- ✅ **2.4** Implement find_optimal_batch_size()
- ✅ **2.5** Add CLI and model constants
- ✅ **2.6** Add torch.compile validation

**Why YOU:** This is the MOST CRITICAL code. Requires deep understanding of PyTorch, gradient computation, memory management, and the core algorithm. The backward pass implementation is technically complex.

---

### Task 3: Flask API Server (All 5 subtasks)
- ✅ **3.1** Create app.py with Flask setup
- ✅ **3.2** Implement POST /analyze endpoint
- ✅ **3.3** Implement GET /health endpoint
- ✅ **3.4** Implement GET /models endpoint
- ✅ **3.5** Add error handling and production config

**Why YOU:** API design, error handling, CORS configuration, and integration with the core engine requires backend expertise.

---

### Task 4: Frontend API Integration (Subtasks 1, 2, 3, 5)
- ✅ **4.1** Create lib/api.ts with typed functions
- ✅ **4.2** Update AnalysisResult interface
- ✅ **4.3** Replace mock data in page.tsx
- ✅ **4.5** Add error handling and user feedback

**Why YOU:** TypeScript, API integration, error handling, and state management require frontend expertise.

---

### Task 5: Backward Pass Validation (All 3 subtasks)
- ✅ **5.1** Implement validate_backward_pass()
- ✅ **5.2** Add CLI flag for validation
- ✅ **5.3** Document validation in README

**Why YOU:** This validates the core technical claim. Requires understanding of gradient computation and memory analysis.

---

### Task 6: Lambda Labs Setup (Subtasks 2, 3, 4, 5)
- ✅ **6.2** Install system dependencies
- ✅ **6.3** Transfer backend code
- ✅ **6.4** Setup Python venv and install packages
- ✅ **6.5** Test backend functionality on GPU

**Why YOU:** SSH, Linux commands, Python environment setup, debugging on remote server.

---

### Task 7: Backend Deployment (All 5 subtasks)
- ✅ **7.1** Configure Flask for production
- ✅ **7.2** Start Flask in tmux session
- ✅ **7.3** Test external API access
- ✅ **7.4** Run integration tests
- ✅ **7.5** Document production API URL

**Why YOU:** Production deployment, tmux, remote debugging, network configuration.

---

### Task 8: Frontend Production Config (Subtasks 1, 3, 4)
- ✅ **8.1** Create environment configuration
- ✅ **8.3** Test production build
- ✅ **8.4** Optimize for Vercel

**Why YOU:** Build optimization, environment variables, production troubleshooting.

---

### Task 9: Vercel Deployment (Subtasks 2, 3, 5)
- ✅ **9.2** Import project to Vercel
- ✅ **9.3** Configure environment variables
- ✅ **9.5** Test CORS and cross-origin

**Why YOU:** Deployment configuration, CORS debugging, production troubleshooting.

---

### Task 10: Demo Materials (Subtasks 1, 2)
- ✅ **10.1** Run validation tests
- ✅ **10.2** Create demo script

**Why YOU:** Technical validation and crafting technical narrative requires domain knowledge.

---

## 🟢 TEAMMATE - 15 Supporting Tasks

### Task 4: Frontend API Integration (Subtask 4 only)
- ⏳ **4.4** Add environment variable configuration

**Instructions:**
```bash
# 1. Create a new file in the project root
touch .env.local

# 2. Add this line to the file
NEXT_PUBLIC_API_URL=http://localhost:5000

# 3. Create another file for documentation
touch .env.example

# 4. Add this to .env.example
NEXT_PUBLIC_API_URL=http://your-lambda-labs-ip:5000

# 5. Verify .env.local is in .gitignore
```

**Why TEAMMATE:** Simple file creation and copy-paste work. Clear instructions.

---

### Task 6: Lambda Labs Setup (Subtask 1 only)
- ⏳ **6.1** Launch Lambda Labs A10 instance

**Instructions:**
1. Go to https://lambdalabs.com/service/gpu-cloud
2. Click "Sign Up" or "Sign In"
3. Click "Launch Instance"
4. Select: **1x A10 (24 GB)**
5. Select OS: **Ubuntu 22.04 LTS + PyTorch**
6. Setup SSH: Follow on-screen instructions
7. Click "Launch"
8. **COPY THE IP ADDRESS** and send it to YOU
9. Test SSH: `ssh ubuntu@<IP_ADDRESS>`

**Why TEAMMATE:** Point-and-click web interface. No coding required. YOU will handle everything after this.

---

### Task 8: Frontend Production Config (Subtasks 2, 5)
- ⏳ **8.2** Update site metadata for MemoryMark branding
- ⏳ **8.5** Create deployment documentation

**Instructions (8.2):**
1. Open file: `app/layout.tsx`
2. Find line with `title:`
3. Replace with: `title: 'MemoryMark - GPU Memory Waste Detector'`
4. Find line with `description:`
5. Replace with: `description: 'Find and fix GPU memory waste in ML training. Analyze BERT, GPT-2, ResNet models. 2-3x speedup, $47k saved.'`
6. Save file
7. Test: Run `npm run dev` and check browser tab title

**Instructions (8.5):**
Create `DEPLOYMENT.md` with this content (YOU will review and expand):
```markdown
# MemoryMark Deployment Guide

## Prerequisites
- Vercel account
- Lambda Labs account
- GitHub account

## Environment Variables
- `NEXT_PUBLIC_API_URL`: Points to Lambda Labs backend

## Deployment Steps
1. Backend deployment: [YOU will fill this in]
2. Frontend deployment: [YOU will fill this in]

## Troubleshooting
- If API doesn't work: Check Lambda Labs instance is running
- If CORS errors: [YOU will fill this in]
```

**Why TEAMMATE:** Simple text editing and documentation skeleton. YOU will complete the technical parts.

---

### Task 9: Vercel Deployment (Subtasks 1, 4, 6)
- ⏳ **9.1** Prepare GitHub repository
- ⏳ **9.4** Deploy and test live site
- ⏳ **9.6** Test mobile responsiveness

**Instructions (9.1):**
```bash
# Verify .gitignore has these lines (add if missing)
node_modules/
.next/
.env.local
.DS_Store
backend/venv/
__pycache__/

# Commit and push
git add .
git commit -m "Prepare repository for Vercel deployment"
git push origin main
```

**Instructions (9.4):**
1. After YOU configure Vercel, click the "Deploy" button
2. Wait for build (watch the logs)
3. When done, click the deployment URL
4. Test: Select "BERT", click "Analyze Model"
5. Screenshot the results page
6. Report to YOU: "Deployment worked" or "Got error: [paste error]"

**Instructions (9.6):**
1. Open live site on your phone OR use Chrome DevTools mobile view
2. Test on phone:
   - Can you read all text?
   - Can you click all buttons?
   - Do the charts/gauges display?
   - Does analysis work?
3. Take screenshots of any issues
4. Report results to YOU

**Why TEAMMATE:** Mechanical tasks (git commands, clicking deploy button, testing). No complex debugging required. Any issues will be escalated to YOU.

---

### Task 10: Demo Materials (Subtasks 3, 4, 5)
- ⏳ **10.3** Practice demo and record backup video
- ⏳ **10.4** Create demo day slides
- ⏳ **10.5** Create checklist and backup plans

**Instructions (10.3):**
1. After YOU write the demo script, practice presenting it 5 times
2. Time yourself each run
3. Record screen + audio using:
   - Mac: QuickTime Player → File → New Screen Recording
   - Windows: Win+G (Game Bar)
   - Linux: OBS Studio
4. Record: Show live site, run analysis, explain results (read from script)
5. Keep video under 3 minutes
6. Send video file to YOU for review

**Instructions (10.4):**
Create 5 slides in Google Slides:
1. **Title Slide**: "MemoryMark" + tagline "GPU Memory Waste Detector"
2. **Problem Slide**: "60-70% GPU Memory Waste" (big text, simple graphic)
3. **Solution Slide**: "Full Training Simulation" (diagram YOU will provide)
4. **Impact Slide**: "$47,000 saved • 3x speedup" (big numbers)
5. **Tech Slide**: "PyTorch • Next.js • Lambda Labs A10"

Use dark theme. Make text BIG and readable. Export as PDF.

**Instructions (10.5):**
Create `DEMO_DAY_CHECKLIST.md`:
```markdown
# Demo Day Checklist

## 30 Minutes Before
- [ ] Start Lambda Labs instance
- [ ] Test backend: curl http://<IP>:5000/health
- [ ] Test live site: Run one analysis
- [ ] Open slides in presentation mode
- [ ] Have backup video file ready
- [ ] Charge laptop to 100%
- [ ] Test phone hotspot

## Backup Plans
- Plan A: Live demo
- Plan B: Use pre-tested responses (if network slow)
- Plan C: Show terminal demo
- Plan D: Play backup video

## Likely Questions
[YOU will fill this section]
```

**Why TEAMMATE:** Presentation practice, slide creation, and checklist organization. Builds confidence and ownership without requiring deep technical knowledge.

---

## 📊 Task Summary

| Role | Task Count | Complexity | Risk Level |
|------|-----------|------------|-----------|
| **YOU** | 35 tasks | High | Critical path |
| **TEAMMATE** | 15 tasks | Low-Medium | Non-blocking |

---

## 🎯 Teammate's Learning Opportunities

Your teammate will gain experience with:
- Git workflow (commits, pushes)
- Environment variables and configuration
- Cloud platform interfaces (Lambda Labs, Vercel)
- Deployment processes
- Testing and QA mindset
- Technical documentation
- Presentation skills

---

## ⚠️ Important Notes for YOU

1. **Review teammate's work before merging** - Especially .gitignore and environment configs
2. **Teammate escalates all errors to YOU** - Don't let them get stuck debugging
3. **Pair program on Task 6.1** - Lambda Labs setup is important, walk through it together
4. **Review slides and video** - Ensure demo materials meet your quality standards
5. **YOU own all critical path items** - Backend engine, API integration, deployment configuration

---

## 🚀 Suggested Workflow

### Week 1: Backend Foundation (YOU)
- Tasks 1, 2, 3, 5
- TEAMMATE: Work on 4.4 (env files)

### Week 2: Infrastructure (YOU + TEAMMATE)
- YOU: Tasks 6.2-6.5, 7
- TEAMMATE: Task 6.1 (launch instance), start on 8.2

### Week 3: Frontend Integration (YOU)
- Tasks 4.1, 4.2, 4.3, 4.5, 8.1, 8.3, 8.4
- TEAMMATE: Task 8.5 (documentation), 9.1 (git prep)

### Week 4: Deployment (YOU + TEAMMATE)
- YOU: Tasks 9.2, 9.3, 9.5
- TEAMMATE: Tasks 9.4, 9.6

### Week 5: Demo Prep (BOTH)
- YOU: Tasks 10.1, 10.2
- TEAMMATE: Tasks 10.3, 10.4, 10.5

---

## 📞 Communication Protocol

**Teammate should message YOU when:**
- Starting a new task
- Encountering any error or uncertainty
- Completing a task for review
- Before pushing any code

**YOU should check in with teammate:**
- Daily: "What task are you working on?"
- After teammate completes a task: Quick review
- Before critical deployments: Pair programming session

---

## ✅ Success Criteria

**For TEAMMATE:**
- All 15 tasks completed without breaking anything
- Learned git, deployment, and testing workflows
- Gained confidence in technical environments
- Delivered quality demo materials

**For YOU:**
- Core engine works perfectly (backward pass validated)
- API integration functional
- Production deployment stable
- Technical claims accurate and demonstrable
