# NourishAI — Smart Recipe & Meal Plan Recommender v5

A beautiful, AI-powered meal planning app built with Streamlit.

## ✨ What's New in v5 (Enhanced)
- **BMR Calculator** — Mifflin–St Jeor formula estimates your daily calorie need from weight, height, age & sex
- **Goal-adjusted targets** — BMR multiplied by goal modifier (e.g. 0.85× for Weight Loss)
- **5 summary metrics** — added Avg Protein alongside the original 4
- **Nutrition Overview tab** — macro breakdown + per-cuisine calorie chart
- **Smarter match badge** — colour-coded green / orange / grey by match strength
- **Improved sidebar layout** — clearer section icons, BMR summary widget

## ML Techniques
| Model | Type | Purpose |
|---|---|---|
| K-Means Clustering | Unsupervised | Group users into dietary personas |
| SVD Matrix Factorization | Collaborative Filtering | Learn latent taste preferences |
| Cosine Similarity | Content-Based | Match recipe features to user profile |
| Weighted Hybrid Fusion | Ensemble | Combine all three signals |

## Project Structure
```
NourishAI/
├── app.py                  # Streamlit app (main entry point)
├── recommender_engine.py   # ML logic & data generation
├── requirements.txt
├── runtime.txt
└── README.md
```

## 🚀 Running Locally (VS Code)

### 1. Clone / download this folder into VS Code

### 2. Create a virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```
The app opens at **http://localhost:8501**

## 🌐 Deploying to Streamlit Community Cloud

1. Push the project to a **public GitHub repo**
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo, branch `main`, and set **Main file path** to `app.py`
4. Click **Deploy** — done! 🎉

No secrets or environment variables are needed.

## Features
- Personalised recipe recommendations based on diet, goals & allergens
- Hybrid ML engine (collaborative + content-based + clustering)
- 7-day meal plan with nutritional summaries
- YouTube video links for every recipe
- Beautiful dark food-themed UI with Playfair Display typography
- Calorie & prep-time filtering
- Cuisine and meal-type filtering
- BMR / daily calorie estimation
- Nutrition Overview tab with macro & cuisine charts
