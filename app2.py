import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Set Streamlit page config
st.set_page_config(page_title="🧠 Medicare Fraud Detector", layout="wide")

# Styled Title
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Medicare Provider Fraud Detection System</h1>", unsafe_allow_html=True)
st.markdown("---")

# Load Model
@st.cache_resource
def load_model():
    return joblib.load("fraud_detection_model.pkl")

model = load_model()

# Sidebar Upload
# Sidebar Upload + Settings
st.sidebar.header("📁 Upload CSV File")
uploaded_file = st.sidebar.file_uploader("Choose a prediction-ready CSV file", type="csv")

st.sidebar.markdown("---")
threshold = st.sidebar.slider("⚙️ Prediction Threshold (%)", 0, 100, 50)

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# Main Logic
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("📄 Preview of Uploaded Data")
        st.dataframe(df.head())

        # Drop non-numeric columns not used by the model
        input_df = df.drop(columns=['Provider', 'State'], errors='ignore')

        # Predict
        if st.button("🚀 Predict Fraud"):
            # Already defined in sidebar, so just use it


            with st.spinner("Running predictions..."):
                probas = model.predict_proba(input_df)
                df['Score'] = [round(p[1] * 100, 2) for p in probas]
                df['Prediction'] = [1 if p[1] > threshold / 100 else 0 for p in probas]

                fraud_count = df['Prediction'].sum()
                non_fraud_count = len(df) - fraud_count

            # Success message
            st.success(f"✅ Prediction complete! 🚨 {fraud_count} suspected fraud cases detected at {threshold}% threshold.")

            # Full Results
            st.subheader("🔍 Prediction Results")
            st.dataframe(df)

            # Score Histogram
            with st.expander("📈 Score Distribution"):
                fig_score = px.histogram(df, x="Score", nbins=20, title="Fraud Probability Score Distribution")
                st.plotly_chart(fig_score, use_container_width=True)

            # Chart: Fraud vs Not Fraud
            with st.expander("📊 View Fraud Summary Chart", expanded=True):
                chart_data = pd.DataFrame({
                    'Type': ['Fraud', 'Not Fraud'],
                    'Count': [fraud_count, non_fraud_count]
                })
                fig = px.bar(chart_data, x='Type', y='Count', color='Type', height=400,
                             color_discrete_map={'Fraud': '#EF553B', 'Not Fraud': '#00CC96'})
                st.plotly_chart(fig, use_container_width=True)

            # Download Button
            st.subheader("⬇️ Download Predictions")
            csv = convert_df_to_csv(df)
            st.download_button(
                label="📥 Download results as CSV",
                data=csv,
                file_name='fraud_predictions.csv',
                mime='text/csv'
            )


            # State-Level Analysis
            if 'State' in df.columns:
                st.markdown("## 🗺️ State-Level Fraud Risk Analysis")

                fraud_by_state = df.groupby('State').agg(
                    total_claims=('Prediction', 'count'),
                    fraud_cases=('Prediction', 'sum')
                ).reset_index()
                fraud_by_state['Fraud Rate (%)'] = round((fraud_by_state['fraud_cases'] / fraud_by_state['total_claims']) * 100, 2)

                cleanest_states = fraud_by_state.sort_values(by='Fraud Rate (%)').head(3)

                st.markdown("### 🏆 Top 3 Cleanest States (Lowest Fraud Rates)")
                st.table(cleanest_states[['State', 'Fraud Rate (%)']])

                st.markdown("### 📊 Fraud Rate by State")
                st.dataframe(fraud_by_state.sort_values(by='Fraud Rate (%)'))

                with st.expander("📈 Visualize Fraud Rate by State"):
                    fig_state = px.bar(
                        fraud_by_state.sort_values(by='Fraud Rate (%)'),
                        x='State', y='Fraud Rate (%)',
                        title="Fraud Rate by State",
                        color='Fraud Rate (%)',
                        color_continuous_scale='greens_r'
                    )
                    st.plotly_chart(fig_state, use_container_width=True)
            else:
                st.info("📍 State information not found in your uploaded file. Add 'State' column to enable regional analysis.")

    except Exception as e:
        st.error("⚠️ Failed to process the file. Ensure it matches the expected format.")
        st.exception(e)
else:
    st.info("⬅️ Upload a processed CSV file from the sidebar to begin.")
