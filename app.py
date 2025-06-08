import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

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
    "melanin_incontinence", "eosinophils_in_the_infiltrate", "PNL_infiltrate",
    "fibrosis_papillary_dermis", "exocytosis", "acanthosis", "hyperkeratosis",
    "parakeratosis", "clubbing_rete_ridges", "elongation_rete_ridges",
    "thinning_suprapapillary_epidermis", "spongiform_pustule", "munro_microabcess",
    "focal_hypergranulosis", "disappearance_granular_layer",
    "vacuolisation_damage_basal_layer", "spongiosis", "saw_tooth_appearance_retes",
    "follicular_horn_plug", "perifollicular_parakeratosis",
    "inflammatory_monoluclear_inflitrate", "band_like_infiltrate", "age", "class"
]

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("dermatology.data", header=None, names=column_names)
    df.replace("?", pd.NA, inplace=True)
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df.dropna(inplace=True)
    df["class"] = df["class"].astype(int)
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
    st.subheader("📊 Dataset Summary")
    st.dataframe(df.describe())

elif option == "Graphs":
    st.subheader("📊 Class Distribution")
    st.bar_chart(df['class'].value_counts())

elif option == "Predict":
    st.subheader("🌳 Disease Prediction using Decision Tree")

    X = df.drop("class", axis=1)
    y = df["class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    st.markdown("### 📈 Model Evaluation")
    st.write("**Accuracy:**", accuracy_score(y_test, y_pred))
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    st.markdown("### 🔍 Try Custom Prediction")
    with st.form("predict_form"):
        input_features = []
        for col in X.columns:
            val = st.number_input(f"{col}", min_value=0, max_value=10, value=1)
            input_features.append(val)

        submitted = st.form_submit_button("Predict Disease Class")
        if submitted:
            result = clf.predict([input_features])
            st.success(f"Predicted Disease Class: {result[0]}")

st.markdown("---")
st.markdown("💡 Dataset Source: [UCI Dermatology Dataset](https://archive.ics.uci.edu/ml/datasets/Dermatology)")

