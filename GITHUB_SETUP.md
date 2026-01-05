# Create GitHub Repository - Step by Step

## Step 1: Install Git

**Windows:** Download from https://git-scm.com/download/win

**Linux:**
```bash
sudo apt install git -y
```

**macOS:**
```bash
xcode-select --install
```

Verify: `git --version`

---

## Step 2: Configure Git

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

---

## Step 3: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `satellite-image-segmentation`
3. Description: `Deep learning pipeline for satellite and aerial image segmentation`
4. **Do NOT** check any boxes (README, .gitignore, license)
5. Click **Create repository**

---

## Step 4: Extract and Navigate to Files

```bash
# Extract the zip file
# Navigate to the folder
cd path/to/satellite-image-segmentation
```

---

## Step 5: Initialize and Push

```bash
# Initialize git
git init

# Add all files
git add .

# First commit
git commit -m "Initial commit: Satellite image segmentation pipeline"

# Connect to GitHub (replace YOUR_USERNAME)
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/satellite-image-segmentation.git

# Push
git push -u origin main
```

---

## Step 6: Authentication

When asked for password, use a **Personal Access Token**:

1. Go to https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Select `repo` scope
4. Copy the token
5. Paste as password

---

## Quick Copy-Paste (Replace YOUR_USERNAME)

```bash
cd path/to/satellite-image-segmentation
git init
git add .
git commit -m "Initial commit: Satellite image segmentation pipeline"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/satellite-image-segmentation.git
git push -u origin main
```

---

## Future Updates

```bash
git add .
git commit -m "Description of changes"
git push
```

---

## Add Topics (Optional)

On your GitHub repo page, click ⚙️ and add topics:
- `deep-learning`
- `pytorch`
- `satellite-imagery`
- `semantic-segmentation`
- `remote-sensing`
- `computer-vision`
