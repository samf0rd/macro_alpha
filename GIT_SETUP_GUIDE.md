# 🔧 Git Setup Guide

## Initial Setup (First Time Only)

### 1. Initialize Git Repository

```powershell
# Make sure you're in the macro_alpha folder
cd C:\Users\Sam Garcia\PycharmProjects\macro_alpha

# Initialize git
git init

# Add all files (respects .gitignore)
git add .

# First commit
git commit -m "Initial commit: Project structure and ETL pipeline"
```

### 2. Connect to GitHub

```powershell
# Create a new repository on GitHub first:
# Go to https://github.com/samf0rd/macro_alpha
# Click "New repository"
# Name it: macro_alpha
# DON'T initialize with README (we already have one)

# Link your local repo to GitHub
git remote add origin https://github.com/samf0rd/macro_alpha.git

# Push to GitHub
git branch -M main
git push -u origin main
```

---

## Daily Git Workflow

### Recommended Commit Structure

```
Day 1: Data Pipeline
Day 2: Feature Engineering
Day 3: Stationarity Testing
Day 4: Baseline Model
...
```

### Example Workflow

```powershell
# 1. Check what changed
git status

# 2. Stage specific files
git add src/data_pipeline.py
git add README.md

# 3. Commit with descriptive message
git commit -m "feat: Add ETL pipeline with FRED API integration"

# 4. Push to GitHub
git push
```

---

## Git Best Practices for This Project

### ✅ DO Commit:
- Python scripts in `src/`
- Jupyter notebooks in `notebooks/`
- README files
- Requirements.txt
- .gitignore
- Config files
- Test files

### ❌ DON'T Commit:
- Data files (`.csv`, `.parquet`) → Too large, regenerate with script
- Model files (`.pkl`, `.h5`) → Too large, retrain
- `venv/` folder → Each person creates their own
- Log files → Generated during execution
- API keys or secrets → Security risk!

---

## Commit Message Conventions

Use these prefixes for clarity:

```
feat: New feature (e.g., "feat: Add RSI indicator calculation")
fix: Bug fix (e.g., "fix: Handle missing FRED data correctly")
docs: Documentation (e.g., "docs: Update README with setup instructions")
refactor: Code cleanup (e.g., "refactor: Extract data cleaning into separate function")
test: Add tests (e.g., "test: Add unit test for feature engineering")
chore: Maintenance (e.g., "chore: Update requirements.txt")
```

### Good Examples:
```
✅ git commit -m "feat: Add walk-forward validation logic"
✅ git commit -m "fix: Prevent look-ahead bias in macro data merge"
✅ git commit -m "docs: Document feature engineering rationale"
```

### Bad Examples:
```
❌ git commit -m "updated stuff"
❌ git commit -m "fixes"
❌ git commit -m "asdf"
```

---

## Branching Strategy (Optional for Solo Project)

If you want to keep things organized:

```powershell
# Create feature branch
git checkout -b feature/mlflow-tracking

# Work on your feature...
git add .
git commit -m "feat: Integrate MLflow experiment tracking"

# Merge back to main
git checkout main
git merge feature/mlflow-tracking

# Delete feature branch
git branch -d feature/mlflow-tracking
```

For this project, working directly on `main` is fine since it's solo.

---

## Useful Git Commands

```powershell
# See commit history
git log --oneline

# Undo last commit (but keep changes)
git reset --soft HEAD~1

# Discard all local changes (CAREFUL!)
git reset --hard HEAD

# See what changed in a file
git diff src/data_pipeline.py

# Pull latest from GitHub
git pull

# Check remote URL
git remote -v
```

---

## GitHub README Features to Use

### Add Badges (Optional but Cool)

At the top of your README.md, add:

```markdown
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Status](https://img.shields.io/badge/status-in%20development-yellow)
![License](https://img.shields.io/badge/license-MIT-green)
```

### Add Images

Store images in `docs/images/` and reference them:

```markdown
![Architecture Diagram](docs/images/architecture.png)
```

---

## 🚨 Important Reminders

1. **NEVER commit sensitive data:**
   - API keys
   - Passwords
   - Personal information
   - Large datasets (> 100MB)

2. **Use .gitignore properly:**
   - Already set up for you
   - Prevents accidental commits of data/models

3. **Commit frequently:**
   - Better to have small, logical commits
   - Easier to track changes and debug

4. **Write clear commit messages:**
   - Future you (and recruiters) will thank you

---

## Sample First Day Commits

```powershell
# After setting up structure
git add .
git commit -m "chore: Initialize project structure with folders and config files"
git push

# After creating ETL script
git add src/data_pipeline.py
git commit -m "feat: Add ETL pipeline for market and macro data ingestion"
git push

# After first successful data pull
git add logs/data_pipeline.log
git commit -m "docs: Add sample execution log from first data pull"
git push
```

---

Ready to initialize your repo? Just follow Step 1 above! 🚀
