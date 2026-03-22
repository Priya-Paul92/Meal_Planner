"""NourishAI — Smart Recipe & Meal Plan Recommender  v5 (Enhanced)"""

import streamlit as st
import pandas as pd
import numpy as np
import time

st.set_page_config(
    page_title="NourishAI — Smart Meal Planner",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from recommender_engine import (
    generate_recipes, generate_users, generate_ratings,
    train_models, recommend_for_new_user, build_meal_plan,
    CUISINES, DIET_TYPES, HEALTH_GOALS, ALLERGENS, DAYS,
    PERSONA_NAMES, INDIAN_REGIONS, get_relevant_allergens,
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #080503 !important; }
.main .block-container { padding: 1.8rem 2.4rem; max-width: 1240px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0D0805 0%,#190D06 55%,#100804 100%) !important;
    border-right: 1px solid rgba(210,100,30,0.18) !important;
}
[data-testid="stSidebar"] * { color: #F5E6D8 !important; }
[data-testid="stSidebar"] label {
    color: rgba(245,230,216,0.60) !important;
    font-size: 0.76rem;
    text-transform: uppercase;
    letter-spacing: .09em;
    font-weight: 500;
}
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stMultiSelect > div > div {
    background: rgba(255,255,255,0.05) !important;
    border-color: rgba(210,100,30,0.28) !important;
    border-radius: 8px !important;
}

/* ── Primary button ── */
.stButton > button {
    background: linear-gradient(135deg,#C45A10,#8B3A08) !important;
    color: #FFF0D8 !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 11px 28px !important;
    transition: opacity .2s, transform .15s !important;
    box-shadow: 0 4px 18px rgba(196,90,16,0.38) !important;
}
.stButton > button:hover {
    opacity: 0.90 !important;
    transform: translateY(-1px) !important;
}

/* ── Metric cards ── */
div[data-testid="stMetric"] {
    background: linear-gradient(145deg,#181008,#201408);
    border: 1px solid rgba(210,100,30,0.22);
    border-radius: 14px;
    padding: 16px 18px;
}
div[data-testid="stMetric"] label { color: rgba(245,230,216,0.50) !important; }
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #FF9A50 !important;
    font-family: 'Playfair Display', serif;
    font-size: 1.7rem !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.04);
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: rgba(245,230,216,0.52) !important;
    font-size: 0.88rem !important;
    border-radius: 8px !important;
    padding: 8px 18px !important;
    transition: all .2s !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(210,100,30,0.28) !important;
    color: #FFB878 !important;
}

/* ── Slider thumb color ── */
[data-testid="stSlider"] > div > div > div > div {
    background: #C45A10 !important;
}

/* ── Progress bars ── */
.stProgress > div > div > div { background: #C45A10 !important; }

/* ── Divider ── */
hr { border-color: rgba(210,100,30,0.15) !important; }

/* ── Toast / info ── */
.stAlert { border-radius: 10px !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0A0604; }
::-webkit-scrollbar-thumb { background: rgba(210,100,30,0.35); border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_data_and_models():
    recipes = generate_recipes()
    users   = generate_users(n=300)
    ratings = generate_ratings(users, recipes, n=8000)
    models  = train_models(recipes, users, ratings)
    return recipes, users, ratings, models


# ── Sidebar helpers ────────────────────────────────────────────────────────────
def _sb_section(title, icon=""):
    st.markdown(
        f'<div style="font-family:Playfair Display,serif;font-size:1.02rem;font-weight:600;'
        f'color:#FF9A50;margin:22px 0 10px;padding-bottom:6px;'
        f'border-bottom:1px solid rgba(210,100,30,0.22);">{icon} {title}</div>',
        unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        # Logo
        st.markdown(
            '<div style="text-align:center;padding:18px 0 8px;">'
            '<div style="font-size:2.4rem;filter:drop-shadow(0 0 12px rgba(255,150,50,0.5));">🍽️</div>'
            '<div style="font-family:Playfair Display,serif;font-size:1.35rem;font-weight:700;'
            'color:#FF9A50;margin-top:6px;letter-spacing:-.01em;">NourishAI</div>'
            '<div style="font-size:0.70rem;color:rgba(245,230,216,0.35);text-transform:uppercase;'
            'letter-spacing:.12em;margin-top:2px;">Smart Meal Planner</div>'
            '</div><hr/>',
            unsafe_allow_html=True)

        _sb_section("Your Profile", "👤")
        diet = st.selectbox("Dietary preference", DIET_TYPES, index=0)
        goal = st.selectbox("Health goal", HEALTH_GOALS, index=0)
        sex  = st.selectbox("Biological sex (for BMR)", ["Male", "Female"], index=0)

        _sb_section("Tastes & Restrictions", "🌍")
        fav_cuisines = st.multiselect(
            "Favourite cuisines", options=CUISINES,
            default=["Indian", "Italian"], max_selections=5)

        indian_region = "All Regions"
        if "Indian" in (fav_cuisines or []):
            indian_region = st.selectbox(
                "Indian region (?)", INDIAN_REGIONS, index=0,
                help="Filter Indian recipes by regional style")

        relevant_allergens = get_relevant_allergens(diet)
        allergies = st.multiselect(
            "Allergies / intolerances", options=relevant_allergens, default=[],
            help="Only allergens relevant to your diet are shown")

        _sb_section("Targets", "🎯")
        calorie_target = st.slider("Calories per meal (kcal)", 200, 900, 480, 10)
        max_prep       = st.slider("Max prep time (minutes)",  10,  90,  30,  5)
        age            = st.slider("Your age", 18, 70, 28, 1)
        weight_kg      = st.slider("Weight (kg)",  40, 150, 70, 1)
        height_cm      = st.slider("Height (cm)", 140, 210, 170, 1)

        days_plan = st.selectbox(
            "Meal plan — number of days (?)", [3, 5, 7], index=2,
            help="3 meals per day: Breakfast, Lunch, Dinner")

        # BMR calculation (Mifflin–St Jeor)
        if sex == "Male":
            bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
        else:
            bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161

        goal_multiplier = {
            "Weight Loss": 0.85, "Muscle Gain": 1.10,
            "Maintenance": 1.0,  "Heart Health": 0.95, "Energy Boost": 1.05,
        }
        daily_target = int(bmr * 1.4 * goal_multiplier.get(goal, 1.0))
        per_meal_bmr = daily_target // 3

        st.markdown(
            f'<div style="background:rgba(255,154,80,0.08);border:1px solid rgba(255,154,80,0.22);'
            f'border-radius:10px;padding:10px 14px;margin:8px 0 14px;">'
            f'<div style="font-size:0.70rem;color:rgba(245,230,216,0.45);text-transform:uppercase;letter-spacing:.08em;">Your est. BMR</div>'
            f'<div style="font-family:Playfair Display,serif;font-size:1.30rem;color:#FF9A50;font-weight:600;">{int(bmr):,} kcal/day</div>'
            f'<div style="font-size:0.72rem;color:rgba(245,230,216,0.40);margin-top:2px;">Suggested per meal: <span style="color:#FFB878;">{per_meal_bmr} kcal</span></div>'
            f'</div>',
            unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)
        go = st.button("✨  Get my recommendations", use_container_width=True)

    return {
        "diet_type":      diet,
        "health_goal":    goal,
        "sex":            sex,
        "fav_cuisines":   fav_cuisines if fav_cuisines else ["Indian"],
        "indian_region":  indian_region,
        "allergies":      allergies,
        "calorie_target": calorie_target,
        "max_prep_min":   max_prep,
        "age":            age,
        "weight_kg":      weight_kg,
        "height_cm":      height_cm,
        "days_plan":      days_plan,
        "bmr":            int(bmr),
        "daily_target":   daily_target,
    }, go


# ── Card utilities ─────────────────────────────────────────────────────────────
CUIS_EMOJI = {
    "Italian": "🇮🇹", "Mexican": "🇲🇽", "Japanese": "🇯🇵", "Indian": "🇮🇳",
    "Mediterranean": "🫒", "American": "🍔", "Thai": "🇹🇭", "Chinese": "🥢",
    "French": "🇫🇷", "Middle Eastern": "🧆",
}
MEAL_EMOJI = {"Breakfast": "☀️", "Lunch": "🌤️", "Dinner": "🌙", "Snack": "🍎"}
REG_EMOJI  = {"North India": "🏔️", "South India": "🌴", "East India": "🌿", "West India": "🌊"}

GOAL_COLOR = {
    "Weight Loss":  "#78FFB0",
    "Muscle Gain":  "#FF9A50",
    "Maintenance":  "#78FFEE",
    "Heart Health": "#FF78A0",
    "Energy Boost": "#FFD700",
}

def _stars(v):
    f = int(round(float(v)))
    return (
        '<span style="color:#FFD700;font-size:.85rem;">' + "★" * f + '</span>'
        + '<span style="color:rgba(255,255,255,0.18);font-size:.85rem;">' + "☆" * (5 - f) + '</span>'
    )

def _nutr_bar(label, val, pct, colour):
    v = float(val)
    p = min(float(pct), 100)
    return (
        '<div style="display:flex;justify-content:space-between;font-size:0.67rem;'
        'color:rgba(245,230,216,0.42);margin-bottom:2px;">'
        f'<span>{label}</span><span>{v:.0f}g</span></div>'
        '<div style="background:rgba(255,255,255,0.07);border-radius:4px;height:4px;margin-bottom:6px;">'
        f'<div style="width:{p:.0f}%;height:4px;border-radius:4px;background:{colour};'
        'transition:width .6s ease;"></div></div>'
    )

def recipe_card(r):
    emoji     = CUIS_EMOJI.get(r["cuisine"], "🍴")
    yt        = str(r.get("youtube_url", "#"))
    score_pct = int(float(r.get("score", 0)) * 100)
    p, c, f   = float(r["protein_g"]), float(r["carbs_g"]), float(r["fat_g"])
    total     = p + c + f + 0.001
    nutr      = (
        _nutr_bar("Protein", p, p / total * 100, "#78FFEE") +
        _nutr_bar("Carbs",   c, c / total * 100, "#FFD700") +
        _nutr_bar("Fat",     f, f / total * 100, "#FF9A50")
    )
    region   = str(r.get("region", ""))
    reg_html = ""
    if region:
        re = REG_EMOJI.get(region, "📍")
        reg_html = (
            f'<span style="display:inline-block;background:rgba(255,160,50,0.12);'
            f'border:1px solid rgba(255,160,50,0.28);color:#FFC080;font-size:0.66rem;'
            f'padding:2px 8px;border-radius:50px;margin-left:6px;">{re} {region}</span>'
        )
    meal_em   = MEAL_EMOJI.get(str(r["meal_type"]), "🍴")
    meal_html = (
        f'<span style="display:inline-block;background:rgba(120,180,255,0.10);'
        f'border:1px solid rgba(120,180,255,0.22);color:#A0C8FF;font-size:0.66rem;'
        f'padding:2px 9px;border-radius:50px;margin-left:4px;">{meal_em} {r["meal_type"]}</span>'
    )
    stars_html = _stars(r["avg_rating"])
    cal        = int(r["calories"])
    prep       = int(r["prep_min"])
    rat        = float(r["avg_rating"])
    name       = str(r["name"])

    # Score badge colour
    if score_pct >= 80:
        badge_bg = "rgba(120,255,180,0.14)"; badge_border = "rgba(120,255,180,0.38)"; badge_color = "#78FFB4"
    elif score_pct >= 60:
        badge_bg = "rgba(210,100,30,0.22)";  badge_border = "rgba(210,100,30,0.48)";  badge_color = "#FFB070"
    else:
        badge_bg = "rgba(255,255,255,0.06)"; badge_border = "rgba(255,255,255,0.15)"; badge_color = "#AAA090"

    out = (
        f'<div style="background:linear-gradient(145deg,#191008,#221508);'
        f'border:1px solid rgba(210,100,30,0.22);border-radius:18px;'
        f'padding:18px 18px 15px;margin-bottom:14px;position:relative;'
        f'transition:border-color .25s;">'

        # Match badge
        f'<div style="position:absolute;top:14px;right:14px;background:{badge_bg};'
        f'border:1px solid {badge_border};color:{badge_color};font-size:0.69rem;'
        f'font-weight:700;padding:3px 10px;border-radius:50px;letter-spacing:.04em;">'
        f'Match {score_pct}%</div>'

        # Title
        f'<div style="font-family:Playfair Display,serif;font-size:1.05rem;font-weight:600;'
        f'color:#FFF0E0;margin:0 0 7px;line-height:1.3;padding-right:86px;">{name}</div>'

        # Tags row
        f'<div style="margin-bottom:12px;">'
        f'<span style="display:inline-block;background:rgba(210,100,30,0.14);'
        f'border:1px solid rgba(210,100,30,0.30);color:#FFAA60;font-size:0.66rem;'
        f'font-weight:500;letter-spacing:.06em;text-transform:uppercase;'
        f'padding:2px 9px;border-radius:50px;">{emoji} {r["cuisine"]}</span>'
        + reg_html + meal_html +
        f'</div>'

        # Stats row
        f'<div style="display:flex;gap:18px;flex-wrap:wrap;margin-bottom:12px;">'
        f'<span style="font-size:0.80rem;color:rgba(245,230,216,0.48);">🔥 '
        f'<span style="color:#FFD4A0;font-weight:500;">{cal} kcal</span></span>'
        f'<span style="font-size:0.80rem;color:rgba(245,230,216,0.48);">⏱️ '
        f'<span style="color:#FFD4A0;font-weight:500;">{prep} min</span></span>'
        f'<span style="font-size:0.80rem;">{stars_html} '
        f'<span style="color:#FFD4A0;font-weight:500;">{rat:.1f}</span></span>'
        f'</div>'

        # Nutrition bars
        f'<div style="margin-bottom:14px;">{nutr}</div>'

        # YouTube link
        f'<a href="{yt}" target="_blank" rel="noopener noreferrer" '
        f'style="display:inline-flex;align-items:center;gap:6px;'
        f'background:rgba(200,30,30,0.14);border:1px solid rgba(220,60,60,0.34);'
        f'color:#FF9090;font-size:0.77rem;font-weight:500;padding:6px 14px;'
        f'border-radius:8px;text-decoration:none;transition:opacity .2s;">'
        f'▶ Watch on YouTube</a>'

        f'</div>'
    )
    return out


def render_meal_plan(plan_df):
    for day in [d for d in DAYS if d in plan_df["day"].values]:
        rows    = plan_df[plan_df["day"] == day]
        if rows.empty:
            continue
        day_cal   = int(rows["calories"].sum())
        rows_html = ""
        for _, row in rows.iterrows():
            yt    = str(row.get("youtube_url", "#"))
            reg   = str(row.get("region", ""))
            reg_s = (f' <span style="font-size:0.64rem;color:rgba(255,180,80,0.48);">· {reg}</span>') if reg else ""
            me    = MEAL_EMOJI.get(str(row["meal"]), "🍴")
            rows_html += (
                '<div style="display:flex;align-items:center;gap:10px;padding:8px 0;'
                'border-bottom:1px solid rgba(255,255,255,0.04);">'
                f'<div style="min-width:82px;font-size:0.67rem;font-weight:500;'
                f'letter-spacing:.07em;text-transform:uppercase;'
                f'color:rgba(245,230,216,0.34);">{me} {row["meal"]}</div>'
                f'<div style="flex:1;font-size:0.87rem;color:#F0DCC8;">{row["name"]}{reg_s}'
                f' <a href="{yt}" target="_blank" rel="noopener noreferrer" '
                f'style="font-size:0.67rem;color:#FF8080;text-decoration:none;">▶ YT</a></div>'
                f'<div style="font-size:0.76rem;color:rgba(255,180,80,0.58);white-space:nowrap;">'
                f'{int(row["calories"])} kcal</div>'
                '</div>'
            )
        st.markdown(
            '<div style="background:linear-gradient(145deg,#18100A,#1E130A);'
            'border:1px solid rgba(210,100,30,0.16);border-radius:14px;'
            'padding:16px 20px;margin-bottom:12px;">'
            '<div style="font-family:Playfair Display,serif;font-size:1rem;font-weight:600;'
            f'color:#FF9A50;margin-bottom:10px;padding-bottom:8px;'
            f'border-bottom:1px solid rgba(210,100,30,0.16);">'
            f'{day}<span style="font-size:0.74rem;font-weight:400;'
            f'color:rgba(245,230,216,0.32);margin-left:10px;">Total ~{day_cal} kcal</span>'
            f'</div>{rows_html}</div>',
            unsafe_allow_html=True)


# ── Landing page ───────────────────────────────────────────────────────────────
def render_landing():
    st.markdown(
        '<div style="text-align:center;padding:52px 40px 40px;">'
        '<div style="font-size:3.8rem;margin-bottom:14px;'
        'filter:drop-shadow(0 0 20px rgba(255,150,50,0.6));">🥗</div>'
        '<div style="font-family:Playfair Display,serif;font-size:1.75rem;font-weight:600;'
        'color:#FFF0D8;margin-bottom:10px;">Tell us about your tastes</div>'
        '<div style="font-size:0.93rem;color:rgba(245,230,216,0.44);max-width:420px;'
        'margin:0 auto;line-height:1.75;">'
        'Set your preferences in the sidebar, then hit '
        '<strong style="color:#FF9A50;">Get my recommendations</strong>.'
        '</div></div>',
        unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("🧬", "Hybrid ML Engine",    "SVD · Content similarity · Persona clustering"),
        ("🎯", "Cuisine Hard Filter", "Only your selected cuisines appear in results"),
        ("🗺️", "Regional Indian",     "North · South · East · West India recipes"),
        ("📊", "BMR Calculator",      "Estimates your calorie needs from body stats"),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3, c4], cards):
        col.markdown(
            f'<div style="background:linear-gradient(145deg,#1A1008,#211408);'
            f'border:1px solid rgba(210,100,30,0.18);border-radius:14px;'
            f'padding:22px 16px;text-align:center;height:160px;">'
            f'<div style="font-size:1.85rem;margin-bottom:10px;">{icon}</div>'
            f'<div style="font-family:Playfair Display,serif;font-size:0.94rem;'
            f'font-weight:600;color:#FFD4A0;margin-bottom:5px;">{title}</div>'
            f'<div style="font-size:0.76rem;color:rgba(245,230,216,0.38);line-height:1.65;">{desc}</div>'
            f'</div>',
            unsafe_allow_html=True)


# ── Nutrition Summary Chart (HTML bar chart) ───────────────────────────────────
def render_nutrition_summary(recs):
    avg_p = recs["protein_g"].mean()
    avg_c = recs["carbs_g"].mean()
    avg_f = recs["fat_g"].mean()
    total = avg_p + avg_c + avg_f + 0.001

    def bar(label, val, pct, color, kcal):
        return (
            f'<div style="margin-bottom:14px;">'
            f'<div style="display:flex;justify-content:space-between;font-size:0.78rem;'
            f'color:rgba(245,230,216,0.55);margin-bottom:5px;">'
            f'<span style="color:{color};font-weight:500;">{label}</span>'
            f'<span>{val:.1f}g avg · <span style="color:rgba(245,230,216,0.35);">{pct:.0f}%</span></span>'
            f'</div>'
            f'<div style="background:rgba(255,255,255,0.06);border-radius:6px;height:10px;">'
            f'<div style="width:{min(pct,100):.0f}%;height:10px;border-radius:6px;background:{color};'
            f'box-shadow:0 0 8px {color}55;"></div></div>'
            f'</div>'
        )

    st.markdown(
        '<div style="background:linear-gradient(145deg,#181008,#201408);'
        'border:1px solid rgba(210,100,30,0.20);border-radius:16px;padding:20px 22px;margin-bottom:22px;">'
        '<div style="font-family:Playfair Display,serif;font-size:1rem;font-weight:600;'
        'color:#FFD4A0;margin-bottom:16px;">📊 Average Macros Across Recommendations</div>'
        + bar("Protein", avg_p, avg_p / total * 100, "#78FFEE", avg_p * 4)
        + bar("Carbohydrates", avg_c, avg_c / total * 100, "#FFD700", avg_c * 4)
        + bar("Fat", avg_f, avg_f / total * 100, "#FF9A50", avg_f * 9)
        + '</div>',
        unsafe_allow_html=True)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    # Hero banner
    st.markdown(
        '<div style="background:linear-gradient(135deg,'
        'rgba(10,6,3,0.88) 0%,rgba(90,35,10,0.78) 45%,rgba(10,6,3,0.88) 100%),'
        'url(https://images.unsplash.com/photo-1543353071-873f17a7a088?w=1600&q=80) '
        'center/cover no-repeat;border-radius:22px;padding:56px 46px 52px;margin-bottom:28px;">'
        '<div style="display:inline-flex;align-items:center;gap:6px;'
        'background:rgba(210,100,30,0.24);border:1px solid rgba(210,100,30,0.45);'
        'color:#FFCFA0;font-size:0.73rem;font-weight:500;letter-spacing:.07em;'
        'padding:5px 15px;border-radius:50px;margin-bottom:18px;">'
        '✦ AI-Powered · Personalised · Nutritionally Balanced</div>'
        '<div style="font-family:Playfair Display,serif;font-size:3.1rem;font-weight:700;'
        'color:#FFF8F0;line-height:1.12;margin:0 0 12px;">Your smart<br/>meal planner</div>'
        '<div style="font-size:1.03rem;color:rgba(255,240,220,0.65);font-weight:300;'
        'max-width:500px;line-height:1.70;">'
        'Discover recipes tailored to your diet, taste, and health goals. '
        'Now with regional Indian cuisine &amp; BMR-based calorie targets.'
        '</div></div>',
        unsafe_allow_html=True)

    with st.spinner("Warming up the kitchen… 👨‍🍳"):
        recipes, users, ratings, models = load_data_and_models()

    profile, go = render_sidebar()

    if not go:
        render_landing()
        return

    with st.spinner("Finding your perfect recipes…"):
        time.sleep(0.25)
        recs = recommend_for_new_user(profile, recipes, ratings, models, top_n=48)

    if recs.empty:
        st.warning("⚠️ No recipes matched your filters — try relaxing the calorie range or prep time.")
        return

    persona = str(recs.iloc[0].get("persona", "Explorer"))
    goal_col = GOAL_COLOR.get(profile["health_goal"], "#FF9A50")

    # Persona + goal pill
    col_p, col_g = st.columns([1, 1])
    with col_p:
        st.markdown(
            f'<div style="display:inline-flex;align-items:center;gap:8px;'
            f'background:linear-gradient(135deg,rgba(210,100,30,0.18),rgba(180,70,10,0.14));'
            f'border:1px solid rgba(210,100,30,0.36);color:#FFB878;font-size:0.86rem;'
            f'font-weight:500;padding:9px 20px;border-radius:50px;margin:4px 0 20px;">'
            f'🧑‍🍳 Your culinary persona: <strong>{persona}</strong></div>',
            unsafe_allow_html=True)
    with col_g:
        st.markdown(
            f'<div style="display:inline-flex;align-items:center;gap:8px;'
            f'background:rgba(0,0,0,0.25);border:1px solid rgba(255,255,255,0.10);'
            f'color:{goal_col};font-size:0.86rem;font-weight:500;'
            f'padding:9px 20px;border-radius:50px;margin:4px 0 20px;">'
            f'🎯 Goal: <strong>{profile["health_goal"]}</strong> · '
            f'BMR: <strong>{profile["bmr"]:,} kcal/day</strong></div>',
            unsafe_allow_html=True)

    # Summary metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Recipes found",  len(recs))
    c2.metric("Avg calories",   f'{recs["calories"].mean():.0f} kcal')
    c3.metric("Avg protein",    f'{recs["protein_g"].mean():.0f}g')
    c4.metric("Avg prep time",  f'{recs["prep_min"].mean():.0f} min')
    c5.metric("Cuisines",       recs["cuisine"].nunique())
    st.markdown("<br/>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🍴  Recipe Recommendations", "📅  Meal Plan", "📊  Nutrition Overview"])

    # ── Tab 1: Recipe cards ───────────────────────────────────────────────────
    with tab1:
        st.markdown(
            '<div style="font-family:Playfair Display,serif;font-size:1.48rem;font-weight:600;'
            'color:#FFF0D8;margin:8px 0 18px;">✦ Top picks for you</div>',
            unsafe_allow_html=True)

        bfast  = recs[recs["meal_type"] == "Breakfast"].reset_index(drop=True)
        lunch  = recs[recs["meal_type"] == "Lunch"].reset_index(drop=True)
        dinner = recs[recs["meal_type"] == "Dinner"].reset_index(drop=True)
        snack  = recs[recs["meal_type"] == "Snack"].reset_index(drop=True)

        mt1, mt2, mt3, mt4 = st.tabs([
            f"☀️  Breakfast ({len(bfast)})",
            f"🌤️  Lunch ({len(lunch)})",
            f"🌙  Dinner ({len(dinner)})",
            f"🍎  Snack ({len(snack)})",
        ])

        def _render_cards(df):
            if df.empty:
                st.info("No recipes in this category. Try widening your cuisine selection or calorie range.")
                return
            col_a, col_b = st.columns(2)
            for i, (_, row) in enumerate(df.iterrows()):
                (col_a if i % 2 == 0 else col_b).markdown(
                    recipe_card(row.to_dict()), unsafe_allow_html=True)

        with mt1: _render_cards(bfast)
        with mt2: _render_cards(lunch)
        with mt3: _render_cards(dinner)
        with mt4: _render_cards(snack)

    # ── Tab 2: Meal plan ──────────────────────────────────────────────────────
    with tab2:
        plan_df = build_meal_plan(recs, days=profile["days_plan"])
        if plan_df.empty:
            st.warning("Not enough variety for a full plan. Try widening your cuisine selection.")
        else:
            region_note = ""
            if profile.get("indian_region", "All Regions") != "All Regions":
                region_note = " · " + profile["indian_region"] + " focus"
            st.markdown(
                f'<div style="font-family:Playfair Display,serif;font-size:1.48rem;'
                f'font-weight:600;color:#FFF0D8;margin:8px 0 18px;">'
                f'Your {profile["days_plan"]}-day meal plan'
                f'<span style="font-size:0.82rem;font-weight:400;'
                f'color:rgba(245,230,216,0.38);margin-left:10px;">3 meals/day{region_note}</span></div>',
                unsafe_allow_html=True)

            total_cal = int(plan_df["calories"].sum())
            daily_avg = total_cal // profile["days_plan"]
            daily_target_display = profile["calorie_target"] * 3

            n1, n2, n3, n4 = st.columns(4)
            n1.metric("Total plan calories",  f"{total_cal:,} kcal")
            n2.metric("Daily average",        f"{daily_avg} kcal")
            n3.metric("Your daily target",    f"{daily_target_display} kcal")
            n4.metric("Est. BMR daily need",  f"{profile['daily_target']} kcal")
            st.markdown("<br/>", unsafe_allow_html=True)
            render_meal_plan(plan_df)

    # ── Tab 3: Nutrition overview ─────────────────────────────────────────────
    with tab3:
        st.markdown(
            '<div style="font-family:Playfair Display,serif;font-size:1.48rem;font-weight:600;'
            'color:#FFF0D8;margin:8px 0 18px;">📊 Nutrition at a Glance</div>',
            unsafe_allow_html=True)
        render_nutrition_summary(recs)

        # Per-cuisine breakdown
        st.markdown(
            '<div style="font-family:Playfair Display,serif;font-size:1.05rem;font-weight:600;'
            'color:#FFD4A0;margin:18px 0 12px;">Avg Calories by Cuisine</div>',
            unsafe_allow_html=True)
        cuis_avg = recs.groupby("cuisine")["calories"].mean().sort_values(ascending=False)
        max_cal  = cuis_avg.max()
        cuisine_html = ""
        for cuisine, cal in cuis_avg.items():
            pct = cal / max_cal * 100
            em  = CUIS_EMOJI.get(cuisine, "🍴")
            cuisine_html += (
                f'<div style="margin-bottom:10px;">'
                f'<div style="display:flex;justify-content:space-between;font-size:0.78rem;'
                f'color:rgba(245,230,216,0.52);margin-bottom:4px;">'
                f'<span>{em} {cuisine}</span><span style="color:#FFD4A0;">{cal:.0f} kcal</span></div>'
                f'<div style="background:rgba(255,255,255,0.06);border-radius:5px;height:7px;">'
                f'<div style="width:{pct:.0f}%;height:7px;border-radius:5px;'
                f'background:linear-gradient(90deg,#C45A10,#FF9A50);"></div></div>'
                f'</div>'
            )
        st.markdown(
            '<div style="background:linear-gradient(145deg,#181008,#201408);'
            'border:1px solid rgba(210,100,30,0.20);border-radius:16px;padding:20px 22px;">'
            + cuisine_html + '</div>',
            unsafe_allow_html=True)


if __name__ == "__main__":
    main()
