import streamlit as st
import pandas as pd

st.set_page_config(page_title="DermaScan", layout="wide")
st.title("🧴 DermaScan: Skin Disease Classifier Dataset Viewer")

# Sidebar navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to", ["Home", "Dataset", "Summary", "Graphs", "Predict"])

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
    df.replace("?", pd.NA, inplace=True)
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df.dropna(inplace=True)
    return df

df = load_data()

# Handle different pages
if option == "Home":
    st.subheader("🏠 Welcome to DermaScan")
    st.markdown("This app helps explore the **UCI Dermatology Dataset** and make predictions.")
    st.image("https://upload.wikimedia.org/wikipedia/commons/0/0d/Skin_disease_icon.png", width=200)

elif option == "Dataset":
    st.subheader("📄 Dataset Preview")
    st.dataframe(df)

elif option == "Summary":
    st.subheader("📊 Class Distribution")
    st.bar_chart(df['class'].value_counts())

elif option == "Graphs":
    st.subheader("🔍 Filter by Age and Class")
    min_age, max_age = int(df['age'].min()), int(df['age'].max())
    age_range = st.slider("Select Age Range", min_age, max_age, (min_age, max_age))

    class_options = sorted(df["class"].unique())
    selected_class = st.multiselect("Select Disease Classes", class_options, default=class_options)

    filtered = df[(df["age"] >= age_range[0]) & (df["age"] <= age_range[1]) & (df["class"].isin(selected_class))]

    st.subheader("📌 Filtered Data")
    st.dataframe(filtered)

elif option == "Predict":
    st.subheader("🧠 Predict Disease (Coming Soon)")
    st.info("This section will include a ML model to predict disease based on features.")

st.markdown("---")
st.markdown("💡 Dataset Source: [UCI Dermatology Dataset](https://archive.ics.uci.edu/ml/datasets/Dermatology)")

