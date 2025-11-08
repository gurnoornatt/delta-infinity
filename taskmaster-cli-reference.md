# Task Master CLI Reference

Quick reference for Claude Task Master commands when used with Claude Code.

## Initial Setup

```bash
# Install globally
npm install -g task-master-ai

# Or install locally in project
npm install task-master-ai

# Initialize project
task-master init
```

## Core CLI Commands

### Project Initialization
```bash
task-master init                    # Initialize new project (.taskmaster directory)
```

### PRD & Task Parsing
```bash
task-master parse-prd               # Parse PRD from stdin
task-master parse-prd --input=prd.txt  # Parse from file
```

### Task Viewing
```bash
task-master list                    # Display all tasks
task-master next                    # Show next task to work on
task-master show [ID]               # View specific task (e.g., task-001)
task-master show [ID1,ID2,ID3]      # View multiple tasks (comma-separated)
```

### Task Management
```bash
task-master move                    # Move tasks between statuses (interactive)
task-master set-status --id=task-001 --status=in-progress
task-master generate                # Generate task files
```

### Analysis & Research
```bash
task-master analyze-complexity      # Analyze task complexity
task-master research "[query]"      # Research with optional project context
```

## Claude Code Configuration

Create/update `.taskmaster/config.json`:

```json
{
  "models": {
    "main": {
      "provider": "claude-code",
      "modelId": "sonnet",
      "maxTokens": 64000,
      "temperature": 0.2
    }
  }
}
```

### Available Models
- `sonnet` - Claude Sonnet (recommended)
- `opus` - Claude Opus

### Advanced Config Options

```json
{
  "models": {
    "main": {
      "provider": "claude-code",
      "modelId": "sonnet",
      "maxTurns": 5,
      "permissionMode": "default",
      "allowedTools": ["Read", "Write", "Edit"],
      "disallowedTools": []
    }
  },
  "commandSpecific": {
    "parse-prd": {
      "maxTurns": 3,
      "appendSystemPrompt": "Focus on breaking down into atomic tasks"
    }
  }
}
```

## Token Optimization

Set environment variable to reduce token usage:

```bash
# Core mode (~5K tokens) - 7 essential tools
export TASK_MASTER_TOOLS=core

# Standard mode (~10K tokens) - 15 common tools
export TASK_MASTER_TOOLS=standard

# All mode (~21K tokens) - 36 complete tools
export TASK_MASTER_TOOLS=all
```

## Task Status Categories

- **backlog** - Planned work
- **in-progress** - Active tasks
- **completed** - Finished tasks

## Project Structure

```
.taskmaster/
├── config.json         # Configuration
├── tasks/             # Task definitions
├── templates/         # Task templates
└── research/          # Research cache
```

## Usage Tips

1. **No API key needed** for Claude Code provider
2. Use `parse-prd` to auto-generate tasks from requirements
3. Use `next` to get AI-recommended next task
4. Use `show` with multiple IDs for batch viewing
5. Set `TASK_MASTER_TOOLS=core` to save tokens
6. Research function pulls fresh data on best practices

## Example Workflow

```bash
# 1. Initialize project
task-master init

# 2. Parse requirements
task-master parse-prd --input=requirements.md

# 3. Check tasks
task-master list

# 4. Get next task
task-master next

# 5. View specific task
task-master show task-001

# 6. Update status
task-master set-status --id=task-001 --status=in-progress

# 7. Research if needed
task-master research "Next.js 14 best practices"
```
