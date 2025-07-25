import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ----------------- App Configuration -----------------
st.set_page_config(page_title="Predictive Maintenance", layout="wide")
st.title("🔧 Predictive Maintenance: RUL Estimator")

# ----------------- Model Selection -----------------
model_option = st.selectbox("📦 Choose Model", ["Random Forest", "XGBoost", "LightGBM"])

@st.cache_resource
def load_model(name):
    return joblib.load(f"{name.lower().replace(' ', '_')}_model.pkl")

model = load_model(model_option)

with st.expander("ℹ️ Model Details"):
    st.write(f"**Model:** {model_option}")
    st.write("**Trained on:** 10,000+ engine cycles")
    st.write("**Performance:** MAE ≈ 7.3 | RMSE ≈ 9.5 (Validation Set)")

# ----------------- File Upload -----------------
uploaded_file = st.file_uploader("📤 Upload Sensor Data CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("📊 Raw Sensor Data")
        st.dataframe(df.head())

        required_columns = ['engine_id', 'cycle']
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        sensor_candidates = [col for col in numeric_cols if col not in required_columns]

        if not all(col in df.columns for col in required_columns):
            st.error("CSV must contain columns: 'engine_id' and 'cycle'")
        else:
            # ----------------- Sensor Selection -----------------
            selected_features = st.multiselect("🧪 Select Sensor Features for Prediction", options=sensor_candidates, default=sensor_candidates[:3])

            if len(selected_features) == 0:
                st.warning("⚠️ Please select at least one sensor feature.")
            else:
                # ----------------- Preprocessing -----------------
                if df[selected_features].isnull().any().any():
                    st.warning("⚠️ Missing values detected. Filling with mean.")
                    df[selected_features] = df[selected_features].fillna(df[selected_features].mean())

                # Predict RUL
                features = df[selected_features]
                df['Predicted_RUL'] = model.predict(features)

                # Uptime %
                df['Uptime_Percentage'] = df['Predicted_RUL'] / (df['cycle'] + df['Predicted_RUL']) * 100

                # ----------------- Engine Filtering -----------------
                st.subheader("🔍 Filter by Engine ID")
                unique_engines = df['engine_id'].unique()
                selected_engines = st.multiselect("Select Engines", unique_engines, default=unique_engines[:3])
                filtered_df = df[df['engine_id'].isin(selected_engines)]

                # ----------------- Smoothing -----------------
                apply_smoothing = st.checkbox("📉 Apply Rolling Average to RUL")
                if apply_smoothing:
                    window = st.slider("Rolling Window Size", 1, 20, 5)
                    filtered_df['Smoothed_RUL'] = filtered_df.groupby('engine_id')['Predicted_RUL'].transform(lambda x: x.rolling(window, min_periods=1).mean())
                    fig = px.line(filtered_df, x='cycle', y='Smoothed_RUL', color='engine_id', title="Smoothed RUL Over Time")
                else:
                    fig = px.line(filtered_df, x='cycle', y='Predicted_RUL', color='engine_id', title="RUL Over Time")

                st.subheader("📈 RUL Prediction Trend")
                st.plotly_chart(fig, use_container_width=True)

                # ----------------- Sensor Trends -----------------
                st.subheader("📉 Sensor Trends Over Time")
                for sensor in selected_features:
                    fig_sensor = px.line(filtered_df, x='cycle', y=sensor, color='engine_id', title=f"{sensor} Over Time")
                    st.plotly_chart(fig_sensor, use_container_width=True)

                # ----------------- Boxplot by Engine -----------------
                st.subheader("📦 RUL Distribution by Engine")
                fig_box = px.box(df, x='engine_id', y='Predicted_RUL', title="Boxplot of RUL per Engine")
                st.plotly_chart(fig_box, use_container_width=True)

                # ----------------- Histogram -----------------
                st.subheader("📊 RUL Histogram")
                fig_hist = px.histogram(df, x='Predicted_RUL', nbins=50, title="Predicted RUL Distribution")
                st.plotly_chart(fig_hist, use_container_width=True)

                # ----------------- Correlation -----------------
                st.subheader("🔗 Feature Correlation")
                corr = df[selected_features].corr()
                fig_corr = px.imshow(corr, text_auto=True, title="Sensor Correlations")
                st.plotly_chart(fig_corr, use_container_width=True)

                # ----------------- Feature Importances -----------------
                st.subheader("🧠 Feature Importances")
                try:
                    importances = model.feature_importances_
                    fig_imp = px.bar(x=selected_features, y=importances, labels={'x': 'Feature', 'y': 'Importance'},
                                     title="Model Feature Importances")
                    st.plotly_chart(fig_imp, use_container_width=True)
                except AttributeError:
                    st.info("This model does not support feature importances.")

                # ----------------- Maintenance Alerts -----------------
                st.subheader("🚨 Maintenance Alerts")
                threshold = st.slider("Set RUL Alert Threshold", min_value=5, max_value=100, value=30)
                alerts = df[df['Predicted_RUL'] <= threshold]

                if alerts.empty:
                    st.success("✅ No engine below the failure threshold.")
                else:
                    st.warning(f"⚠️ {len(alerts)} data points with RUL ≤ {threshold}.")
                    st.dataframe(alerts[['engine_id', 'cycle', 'Predicted_RUL']])

                    # Download alerts
                    st.download_button(
                        label="Download Maintenance Alerts as CSV",
                        data=alerts.to_csv(index=False),
                        file_name="maintenance_alerts.csv",
                        mime="text/csv"
                    )

                # ----------------- Download All Results -----------------
                st.subheader("📥 Download All Predictions")
                st.download_button(
                    label="Download All Results as CSV",
                    data=df.to_csv(index=False),
                    file_name="rul_predictions.csv",
                    mime="text/csv"
                )

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")

else:
    st.info("👈 Upload a CSV with columns: ['engine_id', 'cycle'] + sensor features like ['sensor1', 'sensor2', ...]")
