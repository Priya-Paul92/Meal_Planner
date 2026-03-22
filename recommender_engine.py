"""
NourishAI — Recommender Engine  v4
Fixes:
  - fav_cuisines is now a HARD filter (only selected cuisines shown)
  - meal_type derived from recipe name (deterministic)
  - allergens derived from recipe content (no impossible allergens per diet)
  - Indian region filter works correctly
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

CUISINES      = ["Italian","Mexican","Japanese","Indian","Mediterranean",
                 "American","Thai","Chinese","French","Middle Eastern"]
MEAL_TYPES    = ["Breakfast","Lunch","Dinner","Snack"]
DIET_TYPES    = ["Omnivore","Non-Vegetarian","Vegetarian","Vegan","Keto","Paleo","Gluten-Free"]
HEALTH_GOALS  = ["Weight Loss","Muscle Gain","Maintenance","Heart Health","Energy Boost"]
ALLERGENS     = ["Nuts","Dairy","Gluten","Shellfish","Eggs","Soy"]
TAGS          = ["Quick","High-Protein","Low-Carb","Comfort Food",
                 "Meal Prep","Budget","Spicy","Kid-Friendly","Low-Calorie","High-Fiber"]
DAYS          = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
INDIAN_REGIONS = ["All Regions","North India","South India","East India","West India"]

PERSONA_NAMES = {
    0: "Health-Conscious Explorer",
    1: "Plant-Based Enthusiast",
    2: "Busy Weeknight Cook",
    3: "High-Protein Athlete",
    4: "Comfort Food Lover",
}

# Allergens hidden per diet (never show impossible allergens)
DIET_HIDDEN_ALLERGENS = {
    "Vegetarian"     : ["Shellfish"],
    "Vegan"          : ["Shellfish","Dairy","Eggs"],
    "Omnivore"       : [],
    "Non-Vegetarian" : [],
    "Keto"           : [],
    "Paleo"          : [],
    "Gluten-Free"    : [],
}

def get_relevant_allergens(diet_type):
    hidden = DIET_HIDDEN_ALLERGENS.get(diet_type, [])
    return [a for a in ALLERGENS if a not in hidden]


# ── Name-based classifiers ─────────────────────────────────────
MEAT_KW = [
    "chicken","beef","pork","lamb","turkey","duck","bacon","sausage",
    "prawn","shrimp","salmon","tuna","crab","lobster","fish","steak",
    "veal","mutton","keema","nihari","rogan josh","butter chicken",
    "bouillabaisse","poulet","kung pao","char siu","karaage","yakitori",
    "souvlaki","shawarma","kofta","tikka masala","larb","satay",
    # new non-veg additions
    "boiled chicken","chicken boil","fish boil","boiled fish","boiled egg",
    "egg boil","boiled eggs","grilled chicken","grilled fish","fish fry",
    "chicken fry","mutton curry","chicken curry","fish curry","egg curry",
    "chicken biryani","mutton biryani","egg biryani","fish biryani",
    "chicken soup","chicken stew","fish stew","prawn curry","prawn masala",
    "egg masala","andhra chicken","chettinad chicken","kerala fish",
    "fish molee","meen curry","chicken 65","chicken lollipop",
    "tandoori chicken","seekh kebab","galouti kebab","murgh",
    # northeast style
    "northeast boiled","bamboo chicken","assamese fish","nagaland smoked",
    "manipuri boiled","smoked chicken","bamboo stew",
    # egg bhurji
    "egg bhurji","bhurji",
]
DAIRY_EGG_KW = [
    "egg","omelette","quiche","croque","huevos","shakshuka","paneer",
    "cheese","carbonara","caprese","tamago","gohan","rancheros",
    "labneh","madame","lorraine","kheer","payasam","lassi","raita","curd","dahi",
]
BREAKFAST_KW = [
    "upma","idli","dosa","poha","paratha","porridge","overnight oats",
    "pancake","toast","omelette","tamago gohan","congee","smoothie bowl",
    "croque madame","huevos","shakshuka","rancheros","gohan","onigiri",
    "crepe","blueberry","matcha overnight","avocado toast","puttu","pongal",
    "akki roti","pesarattu","uttapam","appam","dhokla","thepla",
    "boiled egg","egg boil","boiled eggs","egg toast","egg paratha",
    "chicken egg","masala omelette","keema paratha",
    "egg bhurji","bhurji paratha","masala egg bhurji",
]
SNACK_KW = [
    "bruschetta","crostini","hummus plate","baba ganoush plate","fattoush",
    "elote corn salad","guacamole","spring roll bowl","edamame",
    "scallion pancake","papaya salad","bhel","samosa","pakora","vada",
    "kachori","tikki","chaat","khandvi","batata vada",
]

def _classify_meal_type(name):
    n = name.lower()
    if any(k in n for k in BREAKFAST_KW):  return "Breakfast"
    if any(k in n for k in SNACK_KW):      return "Snack"
    if any(k in n for k in ["soup","salad","wrap","plate","dal tadka","dal ","rice","curry","stew","shukto",
                              "lentil","rasam","sambhar","shorba","vichyssoise",
                              "minestrone","ribollita","tom yum","onion soup"]):
        return "Lunch"
    return "Dinner"

def _classify_diet(name):
    n = name.lower()
    if any(k in n for k in MEAT_KW):
        tags = ["Omnivore","Paleo"]
        if any(k in n for k in ["steak","beef","bacon","chicken","salmon","tuna"]):
            tags.append("Keto")
        return tags
    if any(k in n for k in DAIRY_EGG_KW):
        return ["Omnivore","Vegetarian"]
    tags = ["Omnivore","Vegetarian","Vegan"]
    if any(k in n for k in ["rice","quinoa","lentil","potato","corn","hummus",
                              "falafel","dal","chana","rajma","edamame","tofu",
                              "aloo","papaya","mango","idli","dosa","upma","poha"]):
        tags.append("Gluten-Free")
    return tags

def _classify_allergens(name, diet_tags):
    n = name.lower()
    out = []
    if "Vegan" not in diet_tags:
        if any(k in n for k in ["paneer","cheese","milk","cream","dahi","curd","raita","kheer","lassi","payasam"]):
            out.append("Dairy")
        if any(k in n for k in ["egg","omelette","tamago","gohan","quiche","lorraine","carbonara"]):
            out.append("Eggs")
    if any(k in n for k in ["pasta","noodle","ramen","udon","soba","wheat","flour","paratha","naan","puri","roti","bread"]):
        out.append("Gluten")
    if any(k in n for k in ["tofu","soy","edamame","miso"]):
        out.append("Soy")
    if any(k in n for k in ["nut","almond","cashew","peanut","walnut","pistachio"]):
        out.append("Nuts")
    if ("Omnivore" in diet_tags and "Vegetarian" not in diet_tags
            and any(k in n for k in ["prawn","shrimp","crab","lobster"])):
        out.append("Shellfish")
    return list(set(out))

def youtube_url(name):
    return "https://www.youtube.com/results?search_query=" + (name+" recipe").replace(" ","+")


# ── Recipe catalog ─────────────────────────────────────────────
RECIPES_BY_CUISINE = {
    "Italian": [
        "Spaghetti Carbonara","Risotto ai Funghi","Penne Arrabbiata",
        "Margherita Flatbread","Caprese Omelette","Gnocchi al Pesto",
        "Bruschetta Bowl","Ribollita Soup","Caponata Crostini",
        "Focaccia with Olives","Pasta e Fagioli","Minestrone Soup",
        # Non-veg
        "Chicken Piccata","Grilled Salmon Pasta","Shrimp Scampi",
        "Chicken Cacciatore","Tuna Pasta Salad",
    ],
    "Mexican": [
        "Black Bean Tacos","Huevos Rancheros","Guacamole Grain Bowl",
        "Elote Corn Salad","Tamale Breakfast Bowl","Frijoles Negros Soup",
        "Veggie Enchiladas","Sopa de Lima","Chilaquiles Verdes",
        "Bean and Cheese Burrito","Quesadilla de Frijoles","Pozole de Verduras",
        # Non-veg
        "Chicken Tacos","Beef Burrito Bowl","Shrimp Tacos",
        "Chicken Enchiladas","Fish Tacos",
    ],
    "Japanese": [
        "Miso Ramen","Tamago Gohan","Tofu Miso Soup",
        "Soba Noodle Salad","Matcha Overnight Oats","Edamame Fried Rice",
        "Vegetable Gyoza","Agedashi Tofu","Mushroom Udon",
        "Avocado Onigiri","Zaru Soba","Vegetable Tempura Bowl",
        # Non-veg
        "Chicken Karaage","Salmon Donburi","Tuna Onigiri",
        "Yakitori Chicken","Prawn Tempura Bowl",
    ],
    "Indian": [
        # North India - Veg
        "Palak Paneer","Dal Makhani","Aloo Paratha",
        "Paneer Butter Masala","Chole Bhature","Rajma Chawal",
        "Sarson da Saag","Kadai Paneer","Matar Paneer",
        "Stuffed Paneer Paratha","Paneer Tikka","Baingan Bharta",
        "Dum Aloo","Dal Tadka","Pindi Chana",
        # North India - Non Veg
        "Butter Chicken","Mutton Rogan Josh","Chicken Biryani",
        "Keema Paratha","Tandoori Chicken","Seekh Kebab",
        "Murgh Makhani","Nihari Mutton","Chicken Tikka Masala",
        "Egg Bhurji","Egg Bhurji Paratha","Masala Egg Bhurji",
        # South India - Veg
        "Masala Dosa","Idli Sambar","Uttapam",
        "Pesarattu","Pongal Breakfast","Rasam Rice",
        "Avial","Kootu Curry","Medu Vada",
        "Appam with Coconut Stew","Curd Rice","Bisibelebath",
        "Puttu Kadala","Tomato Rasam Soup","Vegetable Stew",
        # South India - Non Veg
        "Chettinad Chicken Curry","Kerala Fish Curry","Andhra Chicken Fry",
        "Fish Molee","Meen Curry","Chicken 65",
        "Prawn Masala","Boiled Egg Curry","Fish Biryani",
        # East India - Veg
        "Luchi Aloo Dum","Cholar Dal","Aloo Posto",
        "Mustard Paneer Curry","Mochar Ghonto","Begun Bhaja",
        "Potoler Dolma","Shukto","Chana Dal Fry",
        # East India - Non Veg
        "Mustard Fish Curry","Kosha Mangsho","Chicken Rezala",
        "Prawn Malai Curry","Hilsa Fish Fry","Egg Devil",
        "Northeast Boiled Chicken","Northeast Boiled Fish","Bamboo Chicken Stew",
        "Assamese Fish Boil","Nagaland Smoked Chicken","Manipuri Boiled Egg Curry",
        # West India - Veg
        "Dhokla Breakfast","Thepla","Undhiyu",
        "Pav Bhaji","Misal Pav","Poha Upma",
        "Gujarati Dal","Batata Vada","Khandvi",
        "Sabudana Khichdi","Puran Poli","Varan Bhat",
        # West India - Non Veg
        "Chicken Kolhapuri","Mutton Biryani Hyderabadi","Goan Fish Curry",
        "Prawn Koliwada","Chicken Sukka","Egg Masala",
    ],
    "Mediterranean": [
        "Greek Salad Bowl","Shakshuka","Hummus Plate",
        "Falafel Wrap","Tabbouleh Quinoa","Stuffed Bell Peppers",
        "Tzatziki Bowl","Spanakopita Bake","Fattoush Salad",
        "Baba Ganoush Plate","Lentil Soup","Roasted Vegetable Couscous",
    ],
    "American": [
        "Avocado Toast","Blueberry Pancakes","Kale Caesar Salad",
        "Sweet Potato Chili","Overnight Oats","Veggie Burger",
        "Mac and Cheese","Buffalo Cauliflower","Peanut Butter Smoothie Bowl",
        "Black Bean Soup","Mushroom Quesadilla","Lentil Tacos",
        # Non-veg
        "Grilled Chicken Salad","Boiled Chicken Bowl","Baked Salmon",
        "Chicken Noodle Soup","BBQ Chicken Wrap","Tuna Salad Sandwich",
        "Boiled Eggs on Toast","Scrambled Eggs Breakfast",
    ],
    "Thai": [
        "Pad Thai Veg","Green Curry Tofu","Tom Yum Soup",
        "Mango Sticky Rice","Papaya Salad","Red Curry Tofu",
        "Thai Peanut Noodles","Tofu Basil Stir Fry","Massaman Potato Curry",
        "Coconut Corn Soup","Thai Fried Rice Veg","Thai Glass Noodle Salad",
        # Non-veg
        "Chicken Pad Thai","Prawn Green Curry","Thai Basil Chicken",
        "Fish Larb Salad","Chicken Satay",
    ],
    "Chinese": [
        "Mapo Tofu","Egg Fried Rice","Hot and Sour Soup",
        "Dim Sum Platter","Congee Bowl","Spring Roll Bowl",
        "Broccoli Tofu Stir Fry","Steamed Dumplings","Buddha's Delight",
        "Scallion Pancakes","Wonton Vegetable Soup","Sesame Noodles",
        # Non-veg
        "Kung Pao Chicken","Char Siu Pork","Chicken Wonton Soup",
        "Steamed Fish with Ginger","Prawn Fried Rice",
    ],
    "French": [
        "Croque Madame","French Onion Soup","Ratatouille Bake",
        "Quiche Lorraine","Crepes aux Champignons","Leek Vinaigrette",
        "Vichyssoise Soup","Tarte Flambee","Gratin Dauphinois",
        "Soupe au Pistou","Nicoise Salad Veg","Salade aux Noix",
    ],
    "Middle Eastern": [
        "Shakshuka Deluxe","Falafel Plate","Manakish Flatbread",
        "Lentil Shorba","Mujaddara","Zaatar Egg Toast",
        "Freekeh Salad","Baba Ganoush Plate","Fattet Hummus",
        "Labneh Bowl","Roasted Eggplant Plate","Lentil and Spinach Soup",
    ],
}

INDIAN_REGION_MAP = {
    "Palak Paneer":"North India","Dal Makhani":"North India","Aloo Paratha":"North India",
    "Paneer Butter Masala":"North India","Chole Bhature":"North India","Rajma Chawal":"North India",
    "Sarson da Saag":"North India","Kadai Paneer":"North India","Matar Paneer":"North India",
    "Stuffed Paneer Paratha":"North India","Paneer Tikka":"North India","Baingan Bharta":"North India",
    "Dum Aloo":"North India","Dal Tadka":"North India","Pindi Chana":"North India",
    # North India Non-veg
    "Butter Chicken":"North India","Mutton Rogan Josh":"North India","Chicken Biryani":"North India",
    "Keema Paratha":"North India","Tandoori Chicken":"North India","Seekh Kebab":"North India",
    "Murgh Makhani":"North India","Nihari Mutton":"North India","Chicken Tikka Masala":"North India",
    "Egg Bhurji":"North India","Egg Bhurji Paratha":"North India","Masala Egg Bhurji":"North India",
    # South India Veg
    "Masala Dosa":"South India","Idli Sambar":"South India","Uttapam":"South India",
    "Pesarattu":"South India","Pongal Breakfast":"South India","Rasam Rice":"South India",
    "Avial":"South India","Kootu Curry":"South India","Medu Vada":"South India",
    "Appam with Coconut Stew":"South India","Curd Rice":"South India","Bisibelebath":"South India",
    "Puttu Kadala":"South India","Tomato Rasam Soup":"South India","Vegetable Stew":"South India",
    # South India Non-veg
    "Chettinad Chicken Curry":"South India","Kerala Fish Curry":"South India","Andhra Chicken Fry":"South India",
    "Fish Molee":"South India","Meen Curry":"South India","Chicken 65":"South India",
    "Prawn Masala":"South India","Boiled Egg Curry":"South India","Fish Biryani":"South India",
    # East India Veg
    "Luchi Aloo Dum":"East India","Cholar Dal":"East India","Aloo Posto":"East India",
    "Mustard Paneer Curry":"East India","Mochar Ghonto":"East India","Begun Bhaja":"East India",
    "Potoler Dolma":"East India","Shukto":"East India","Chana Dal Fry":"East India",
    # East India Non-veg
    "Mustard Fish Curry":"East India","Kosha Mangsho":"East India","Chicken Rezala":"East India",
    "Prawn Malai Curry":"East India","Hilsa Fish Fry":"East India","Egg Devil":"East India",
    "Northeast Boiled Chicken":"East India","Northeast Boiled Fish":"East India",
    "Bamboo Chicken Stew":"East India","Assamese Fish Boil":"East India",
    "Nagaland Smoked Chicken":"East India","Manipuri Boiled Egg Curry":"East India",
    # West India Veg
    "Dhokla Breakfast":"West India","Thepla":"West India","Undhiyu":"West India",
    "Pav Bhaji":"West India","Misal Pav":"West India","Poha Upma":"West India",
    "Gujarati Dal":"West India","Batata Vada":"West India","Khandvi":"West India",
    "Sabudana Khichdi":"West India","Puran Poli":"West India","Varan Bhat":"West India",
    # West India Non-veg
    "Chicken Kolhapuri":"West India","Mutton Biryani Hyderabadi":"West India","Goan Fish Curry":"West India",
    "Prawn Koliwada":"West India","Chicken Sukka":"West India","Egg Masala":"West India",
}


def generate_recipes():
    records = []
    rid = 0
    for cuisine, names in RECIPES_BY_CUISINE.items():
        for name in names:
            diet_tags = _classify_diet(name)
            meal_type = _classify_meal_type(name)
            allergens = _classify_allergens(name, diet_tags)
            tags      = list(np.random.choice(TAGS, size=np.random.randint(2,5), replace=False))
            is_vegan  = "Vegan" in diet_tags
            is_keto   = "Keto" in diet_tags
            region    = INDIAN_REGION_MAP.get(name, "") if cuisine == "Indian" else ""
            records.append({
                "recipe_id"  : f"R{rid:03d}",
                "name"       : name,
                "cuisine"    : cuisine,
                "region"     : region,
                "meal_type"  : meal_type,
                "diet_types" : diet_tags,
                "tags"       : tags,
                "allergens"  : allergens,
                "calories"   : int(np.clip(np.random.normal(420,110),150,900)),
                "protein_g"  : round(np.clip(np.random.normal(14 if is_vegan else 26,7),4,55),1),
                "carbs_g"    : round(np.clip(np.random.normal(12 if is_keto else 46,10),4,80),1),
                "fat_g"      : round(np.clip(np.random.normal(26 if is_keto else 12,5),2,48),1),
                "prep_min"   : int(np.random.choice([10,15,20,30,45,60,90],
                                   p=[0.10,0.15,0.25,0.25,0.15,0.07,0.03])),
                "avg_rating" : round(np.random.beta(7,2)*4+1,1),
                "n_ratings"  : int(np.random.lognormal(4.5,1.0)),
                "youtube_url": youtube_url(name),
            })
            rid += 1
    return pd.DataFrame(records)


def generate_users(n=300):
    records = []
    cal_map = {"Weight Loss":350,"Muscle Gain":650,"Maintenance":480,"Heart Health":420,"Energy Boost":500}
    for i in range(n):
        diet = np.random.choice(DIET_TYPES, p=[0.30,0.20,0.20,0.10,0.08,0.06,0.06])
        goal = np.random.choice(HEALTH_GOALS, p=[0.30,0.25,0.25,0.12,0.08])
        records.append({
            "user_id"       : f"U{i:04d}",
            "diet_type"     : diet,
            "health_goal"   : goal,
            "allergies"     : list(np.random.choice(ALLERGENS,size=np.random.randint(0,2),replace=False)),
            "fav_cuisines"  : list(np.random.choice(CUISINES,size=np.random.randint(2,4),replace=False)),
            "calorie_target": int(np.clip(np.random.normal(cal_map[goal],80),250,850)),
            "max_prep_min"  : int(np.random.choice([15,30,45,60],p=[0.25,0.40,0.25,0.10])),
            "age"           : int(np.clip(np.random.normal(34,10),18,70)),
        })
    return pd.DataFrame(records)


def generate_ratings(users, recipes, n=8000):
    records = []
    for _ in range(n):
        u = users.sample(1).iloc[0]
        r = recipes.sample(1).iloc[0]
        if any(a in r["allergens"] for a in u["allergies"]):
            if np.random.random() < 0.85: continue
        if u["diet_type"] in ("Vegetarian","Vegan","Gluten-Free"):
            if u["diet_type"] not in r["diet_types"]: continue
        rating = np.random.normal(r["avg_rating"],0.6)
        if u["diet_type"] in r["diet_types"] or u["diet_type"]=="Omnivore":
            rating += np.random.uniform(0.1,0.4)
        if r["cuisine"] in u["fav_cuisines"]:
            rating += np.random.uniform(0.1,0.5)
        if abs(r["calories"]-u["calorie_target"]) < 150:
            rating += np.random.uniform(0.05,0.25)
        records.append({"user_id":u["user_id"],"recipe_id":r["recipe_id"],
                        "rating":round(np.clip(rating,1.0,5.0),1)})
    return pd.DataFrame(records).drop_duplicates(["user_id","recipe_id"])


def build_recipe_feature_matrix(recipes):
    mlb_diet  = MultiLabelBinarizer()
    mlb_tags  = MultiLabelBinarizer()
    mlb_allrg = MultiLabelBinarizer()
    df = pd.concat([
        pd.DataFrame(StandardScaler().fit_transform(
            recipes[["calories","protein_g","carbs_g","fat_g","prep_min","avg_rating"]]),
            columns=["calories","protein_g","carbs_g","fat_g","prep_min","avg_rating"]),
        pd.DataFrame(mlb_diet.fit_transform(recipes["diet_types"]),
                     columns=[f"diet_{c}" for c in mlb_diet.classes_]),
        pd.DataFrame(mlb_tags.fit_transform(recipes["tags"]),
                     columns=[f"tag_{c.replace(' ','_')}" for c in mlb_tags.classes_]),
        pd.DataFrame(mlb_allrg.fit_transform(recipes["allergens"]),
                     columns=[f"allrg_{c}" for c in mlb_allrg.classes_]),
        pd.get_dummies(recipes["cuisine"], prefix="cuisine"),
        pd.get_dummies(recipes["meal_type"], prefix="meal"),
    ], axis=1).reset_index(drop=True)
    return df


def build_user_feature_matrix(users):
    mlb_a = MultiLabelBinarizer()
    mlb_c = MultiLabelBinarizer()
    df = pd.concat([
        pd.DataFrame(StandardScaler().fit_transform(
            users[["calorie_target","max_prep_min","age"]]),
            columns=["calorie_target","max_prep_min","age"]),
        pd.get_dummies(users["diet_type"], prefix="diet"),
        pd.get_dummies(users["health_goal"], prefix="goal"),
        pd.DataFrame(mlb_a.fit_transform(users["allergies"]),
                     columns=[f"allrg_{c}" for c in mlb_a.classes_]),
        pd.DataFrame(mlb_c.fit_transform(users["fav_cuisines"]),
                     columns=[f"fav_{c}" for c in mlb_c.classes_]),
    ], axis=1).reset_index(drop=True)
    return df


def train_models(recipes, users, ratings):
    recipe_feat = build_recipe_feature_matrix(recipes)
    user_feat   = build_user_feature_matrix(users)
    X = user_feat.values
    feat_cols = list(user_feat.columns)   # ← store for new-user encoding
    best_k, best_s = 3, -1
    for ki in range(3,7):
        km_t = KMeans(n_clusters=ki, random_state=42, n_init=10)
        s    = silhouette_score(X, km_t.fit_predict(X))
        if s > best_s: best_s, best_k = s, ki
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    users = users.copy()
    users["cluster"] = km.fit_predict(X)
    users["persona"] = users["cluster"].map(PERSONA_NAMES).fillna("Explorer")

    uid_list = sorted(ratings["user_id"].unique())
    rid_list = sorted(ratings["recipe_id"].unique())
    u_idx = {u:i for i,u in enumerate(uid_list)}
    r_idx = {r:i for i,r in enumerate(rid_list)}
    sp = csr_matrix((ratings["rating"].values,
                     (ratings["user_id"].map(u_idx), ratings["recipe_id"].map(r_idx))),
                    shape=(len(uid_list), len(rid_list)))
    umeans = np.array(sp.mean(axis=1)).flatten()
    sp_n = sp.copy().astype(float)
    for i,m in enumerate(umeans): sp_n[i] -= m*(sp_n[i]!=0)
    svd = TruncatedSVD(n_components=20, random_state=42)
    uf  = svd.fit_transform(sp_n)
    rf  = svd.components_.T
    sim = cosine_similarity(recipe_feat.values)
    r_map = {r:i for i,r in enumerate(recipes["recipe_id"].tolist())}
    return {"users":users,"km":km,"user_factors":uf,"recipe_factors":rf,
            "user_idx":u_idx,"recipe_ids_svd":rid_list,"user_means":umeans,
            "sim":sim,"r_map":r_map,"feat_cols":feat_cols}


def _minmax(s):
    mn,mx = s.min(),s.max()
    return s*0 if mx==mn else (s-mn)/(mx-mn)

def _encode_user(p, feat_cols):
    """Build a feature vector that matches the training columns exactly."""
    row = {col: 0.0 for col in feat_cols}
    # Numeric (scaled roughly like StandardScaler on training data)
    if "calorie_target" in row: row["calorie_target"] = (p["calorie_target"] - 480) / 80
    if "max_prep_min"   in row: row["max_prep_min"]   = (p["max_prep_min"]   - 30)  / 15
    if "age"            in row: row["age"]             = (p.get("age", 30)   - 34)  / 10
    # Diet one-hot
    diet_col = f'diet_{p["diet_type"]}'
    if diet_col in row: row[diet_col] = 1.0
    # Goal one-hot
    goal_col = f'goal_{p["health_goal"]}'
    if goal_col in row: row[goal_col] = 1.0
    # Cuisine one-hot
    for c in p.get("fav_cuisines", []):
        col = f"fav_{c}"
        if col in row: row[col] = 1.0
    return [row[col] for col in feat_cols]

def _content_score(profile, recipes):
    scores = []
    favs   = set(profile.get("fav_cuisines",[]))
    region = profile.get("indian_region","All Regions")
    for _, r in recipes.iterrows():
        s = 0.0
        if r["cuisine"] in favs: s += 0.6
        if profile["diet_type"]=="Omnivore" or profile["diet_type"] in r["diet_types"]: s += 0.3
        if abs(r["calories"]-profile["calorie_target"]) < 150: s += 0.2
        if (r["cuisine"]=="Indian" and region not in ("All Regions","")
                and r.get("region","")==region): s += 0.3
        scores.append(s)
    return pd.Series(scores, index=recipes["recipe_id"].tolist())


def recommend_for_new_user(profile, recipes, ratings, models, top_n=24):
    users_t = models["users"]
    all_ids = recipes["recipe_id"].tolist()

    cid     = int(models["km"].predict([_encode_user(profile, models["feat_cols"])])[0])
    persona = PERSONA_NAMES.get(cid,"Explorer")

    cu = users_t[users_t["cluster"]==cid]["user_id"].tolist()
    cr = ratings[ratings["user_id"].isin(cu)]
    cl_scores = (cr.groupby("recipe_id")["rating"].mean()
                 .reindex(all_ids).fillna(0)) if not cr.empty \
                else pd.Series(0.0, index=all_ids)

    hybrid = 0.5*_minmax(cl_scores) + 0.5*_minmax(_content_score(profile, recipes))

    # ── Apply all hard filters ─────────────────────────────────
    f = recipes.copy()

    # 1. CUISINE hard filter — only selected cuisines
    favs = profile.get("fav_cuisines",[])
    if favs:
        f = f[f["cuisine"].isin(favs)]

    # 2. Allergen filter
    if profile.get("allergies"):
        f = f[f["allergens"].apply(lambda a: not any(x in a for x in profile["allergies"]))]

    # 3. Calorie range
    cal = profile["calorie_target"]
    f   = f[(f["calories"]>=cal-220)&(f["calories"]<=cal+220)]

    # 4. Prep time
    f = f[f["prep_min"]<=profile["max_prep_min"]]

    # 5. Diet (strict)
    diet = profile["diet_type"]
    if diet == "Vegetarian":
        f = f[f["diet_types"].apply(lambda dt: "Vegetarian" in dt or "Vegan" in dt)]
    elif diet == "Vegan":
        f = f[f["diet_types"].apply(lambda dt: "Vegan" in dt)]
    elif diet == "Non-Vegetarian":
        # Only show recipes that contain meat/fish/egg — exclude pure veg/vegan
        f = f[f["diet_types"].apply(lambda dt: "Vegetarian" not in dt and "Vegan" not in dt)]
    elif diet not in ("Omnivore",):
        f = f[f["diet_types"].apply(lambda dt: diet in dt)]

    # 6. Indian region (only filters the Indian subset; keeps non-Indian cuisines)
    region = profile.get("indian_region","All Regions")
    if region and region!="All Regions" and "Indian" in favs:
        non_indian  = f[f["cuisine"]!="Indian"]
        indian_only = f[(f["cuisine"]=="Indian")&(f["region"]==region)]
        f = pd.concat([non_indian, indian_only])

    valid  = f["recipe_id"].tolist()
    scores = hybrid[hybrid.index.isin(valid)].nlargest(top_n)
    result = f[f["recipe_id"].isin(scores.index)].copy()
    result["score"]   = result["recipe_id"].map(scores).round(4)
    result["persona"] = persona
    return result.sort_values("score", ascending=False)


def build_meal_plan(recs, days=7):
    plan = []
    for mt in ["Breakfast","Lunch","Dinner"]:
        pool = recs[recs["meal_type"]==mt].to_dict("records")
        if len(pool)<days:
            pool = pool + recs[recs["meal_type"]!=mt].to_dict("records")
        for i,day in enumerate(DAYS[:days]):
            if i<len(pool):
                r = pool[i%len(pool)]
                plan.append({"day":day,"meal":mt,
                    "recipe_id":r["recipe_id"],"name":r["name"],
                    "cuisine":r["cuisine"],"region":r.get("region",""),
                    "calories":r["calories"],"protein_g":r["protein_g"],
                    "carbs_g":r["carbs_g"],"fat_g":r["fat_g"],
                    "prep_min":r["prep_min"],"avg_rating":r["avg_rating"],
                    "youtube_url":r["youtube_url"],"score":r["score"]})
    return pd.DataFrame(plan)
