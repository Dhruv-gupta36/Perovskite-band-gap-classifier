# How to Upload This Project to GitHub

---

## Step 1: Create a `.gitignore`

Create a file called `.gitignore` in the project root with this content:

```
__pycache__/
*.pyc
*.pyo
venv/
.env
models/*.pkl
*.egg-info/
.DS_Store
```

> Note: The `models/` folder (with `.pkl` files) is excluded because trained model files are large and generated locally. A user clones the repo and runs `main.py` to generate them.

---

## Step 2: Initialize Git locally

Open a terminal in your project folder and run:

```bash
cd perovskite_ml
git init
git add .
git commit -m "Initial commit: Perovskite band gap classifier with SVM + FastAPI"
```

---

## Step 3: Create the GitHub repository

1. Go to [https://github.com](https://github.com) and log in.
2. Click the **+** icon → **New repository**.
3. Fill in:
   - **Repository name**: `perovskite-band-gap-classifier`
   - **Description**: *"ML classifier to predict metallic vs. semiconducting behavior in perovskite oxides using SVM, PCA, and quantum-chemical features."*
   - Set to **Public**
   - Do **not** initialize with README (you already have one)
4. Click **Create repository**.

---

## Step 4: Push your code

GitHub will show you the commands. Run these:

```bash
git remote add origin https://github.com/<your-username>/perovskite-band-gap-classifier.git
git branch -M main
git push -u origin main
```

---

## Step 5: Final folder structure on GitHub

Your repo should look like this:

```
perovskite-band-gap-classifier/
├── data/
│   └── dataset_excavate.csv
├── src/
│   ├── preprocessing.py
│   ├── eda.py
│   └── model.py
├── api/
│   ├── app.py
│   └── example_request.py
├── main.py
├── requirements.txt
├── README.md
├── INTERVIEW_NOTES.md
└── .gitignore
```

---

## Step 6: Write a good repository description (for GitHub)

In the repo settings or the description field, use:

> **ML classifier to predict metallic vs. semiconducting behavior in perovskite oxides using SVM, PCA, and quantum-chemical descriptors. Trained on 5,152 compounds. Includes FastAPI inference endpoint.**

**Topics/tags to add** (in the repository settings under "Topics"):
`machine-learning`, `materials-science`, `perovskite`, `svm`, `scikit-learn`, `fastapi`, `python`, `classification`, `pca`

---

## Optional: Add a GitHub Actions badge

If you want the README to show a "build passing" badge, create `.github/workflows/test.yml`:

```yaml
name: Run Training Pipeline
on: [push]
jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt
      - run: python main.py
```

Then add this at the top of your README under the title:

```markdown
![CI](https://github.com/<your-username>/perovskite-band-gap-classifier/actions/workflows/test.yml/badge.svg)
```
