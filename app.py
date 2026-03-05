import streamlit as st
import pandas as pd
import joblib
from automl_engine import AutoMLEngine
# -----------------------------
# Futuristic Neon Dashboard CSS
# -----------------------------

st.markdown("""
<style>

body {
    background-color: #0f172a;
}

.stApp {
    background: linear-gradient(135deg,#020617,#0f172a);
    color: white;
}

h1 {
    color: #38bdf8;
    text-align: center;
    font-size: 48px;
    text-shadow: 0 0 15px #38bdf8;
}

h2, h3 {
    color: #22d3ee;
}

div.stButton > button {
    background-color: #0ea5e9;
    color: white;
    border-radius: 10px;
    height: 45px;
    width: 200px;
    font-size: 16px;
    box-shadow: 0 0 10px #38bdf8;
}

div.stButton > button:hover {
    background-color: #0284c7;
}

</style>
""", unsafe_allow_html=True)

st.title("⚡ SmartForge AutoML Platform")
st.markdown("### Build Machine Learning Models Automatically 🚀")

uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("Dataset Preview:")
    st.dataframe(df.head())
    # ------------------------------
    # Dataset Analysis Dashboard
    # ------------------------------

    st.subheader("Dataset Overview")

    st.write("Shape of Dataset:", df.shape)

    st.write("Column Data Types:")
    st.write(df.dtypes)

    st.write("Missing Values in Each Column:")
    st.write(df.isnull().sum())

    st.subheader("Statistical Summary")
    st.write(df.describe())

    # ------------------------------
# Automatic Data Visualization
# ------------------------------

    st.subheader("Data Visualization")

    numeric_columns = df.select_dtypes(include=['int64','float64']).columns

    if len(numeric_columns) > 0:

        selected_column = st.selectbox(
            "Select Column to Visualize",
            numeric_columns
        )

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        df[selected_column].hist(ax=ax)

        ax.set_title(f"Distribution of {selected_column}")

        st.pyplot(fig)

    target_column = st.selectbox(
        "Select Target Column (If Supervised)",
        ["None"] + list(df.columns)
    )

    if st.button("Run AutoML"):

        if target_column == "None":
            target_column = None

        st.subheader("AutoML Training Progress")

        progress_bar = st.progress(0)

        status = st.empty()

        status.write("Initializing AutoML Engine...")

        progress_bar.progress(20)

        status.write("Detecting Problem Type...")

        progress_bar.progress(40)

        status.write("Training Multiple Models...")

        results = AutoMLEngine.run(df, target_column)

        progress_bar.progress(80)

        status.write("Selecting Best Model...")

        progress_bar.progress(100)

        status.write("Training Completed ✅")

        st.write("Results:")
        st.write(results)

        # ------------------------------
        # Confusion Matrix
        # ------------------------------
        if "Confusion Matrix" in results:
            import matplotlib.pyplot as plt
            import seaborn as sns

            st.subheader("Confusion Matrix")

            fig, ax = plt.subplots()
            sns.heatmap(results["Confusion Matrix"], annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")

            st.pyplot(fig)

        # ------------------------------
        # Leaderboard
        # ------------------------------
        # ------------------------------
    # Leaderboard
    # ------------------------------
        if "Leaderboard" in results:
            st.subheader("Model Comparison Leaderboard")

            leaderboard_df = pd.DataFrame(results["Leaderboard"])
            leaderboard_df = leaderboard_df.sort_values(by="Best CV Score", ascending=False)

            st.dataframe(leaderboard_df)

    # ------------------------------
    # Model Performance Chart
    # ------------------------------

            import matplotlib.pyplot as plt

            st.subheader("Model Performance Comparison")

            fig, ax = plt.subplots()

            ax.bar(
                leaderboard_df["Model"],
                leaderboard_df["Best CV Score"]
            )

            ax.set_xlabel("Models")
            ax.set_ylabel("CV Score")
            ax.set_title("Model Performance Comparison")

            st.pyplot(fig)
        # ------------------------------
        # Download Trained Model
# ------------------------------
        st.subheader("Download Trained Model")

        with open("best_supervised_model.pkl", "rb") as file:
            st.download_button(
                label="Download Best Model",
                data=file,
                file_name="best_supervised_model.pkl",
                mime="application/octet-stream"
            )

        # ------------------------------
        # Prediction Section
        # ------------------------------
        st.subheader("Make Prediction")

        if st.checkbox("Enable Prediction"):

            model = joblib.load("best_supervised_model.pkl")

            input_data = {}

            if target_column is not None:
                feature_columns = df.drop(columns=[target_column]).columns
            else:
                feature_columns = df.columns

            for col in feature_columns:
                value = st.text_input(f"Enter {col}")
                input_data[col] = value

            if st.button("Predict"):

                input_df = pd.DataFrame([input_data])

                # convert to numeric if possible
                input_df = input_df.apply(pd.to_numeric, errors="ignore")

                prediction = model.predict(input_df)

                st.success(f"Prediction Result: {prediction[0]}")