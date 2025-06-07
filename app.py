import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="DermaScan", layout="wide")
st.title("🧴 DermaScan: Skin Disease Classifier Dataset Viewer")

# Sidebar navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to", ["Home", "Dataset", "Graphs", "Predict"])

# Column names
column_names = [
    "erythema", "scaling", "definite_borders", "itching", "koebner_phenomenon",
    "polygonal_papules", "follicular_papules", "oral_mucosal_involvement",
    "knee_and_elbow_involvement", "scalp_involvement", "family_history",
    "melanin_incontinence", "eosinophils_in_the_infiltrate",
    "PNL_infiltrate", "fibrosis_papillary_dermis", "exocytosis", "acanthosis",
    "hyperkeratosis", "parakeratosis", "clubbing_rete_ridges",
    "elongation_rete_ridges", "thinning_suprapapillary_epidermis",
    "spongiform_pustule", "munro_microabcess", "focal_hypergranulosis",
    "disappearance_granular_layer", "vacuolisation_damage_basal_layer",
    "spongiosis", "saw_tooth_appearance_retes", "follicular_horn_plug",
    "perifollicular_parakeratosis", "inflammatory_monoluclear_inflitrate",
    "band_like_infiltrate", "age", "class"
]

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("dermatology.data", header=None, names=column_names)
    df.replace("?", np.nan, inplace=True)
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df.dropna(inplace=True)
    df["class"] = df["class"].astype(int)
    return df

df = load_data()

# Home
if option == "Home":
    st.subheader("🏠 Welcome to DermaScan")
    st.markdown("This app helps explore the **UCI Dermatology Dataset** and predict **skin disease class** using **Machine Learning**.")
    st.image("https://upload.wikimedia.org/wikipedia/commons/0/0d/Skin_disease_icon.png", width=200)

# Dataset tab
elif option == "Dataset":
    st.subheader("📄 Dataset Preview")
    st.dataframe(df)

# Graphs tab
elif option == "Graphs":
    st.subheader("📊 Class Distribution")
    st.bar_chart(df["class"].value_counts())

# Predict tab
elif option == "Predict":
    st.subheader("🧠 Predict Skin Disease Class")

    # Prepare features and labels
    X = df.drop(columns=["class"])
    y = df["class"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Show accuracy
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.success(f"🔍 Model Accuracy: {acc:.2f}")

    st.markdown("### Enter Symptoms to Predict Disease Class")

    # Create inputs dynamically for each feature
    user_input = []
    for col in X.columns:
        val = st.slider(f"{col}", int(df[col].min()), int(df[col].max()), int(df[col].mean()))
        user_input.append(val)

    # Prediction
    input_array = np.array(user_input).reshape(1, -1)
    predicted_class = model.predict(input_array)[0]

    st.success(f"✅ Predicted Disease Class: **{predicted_class}**")

# Footer
st.markdown("---")
st.markdown("📘 Dataset Source: [UCI Dermatology Dataset](https://archive.ics.uci.edu/ml/datasets/Dermatology)")
