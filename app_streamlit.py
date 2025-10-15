import streamlit as st
import requests
import pandas as pd
import sqlite3, os

API_LOCAL = "http://127.0.0.1:5000/api/symptom-check"
DB_PATH = "/content/health-symptom-checker/history.db"

st.set_page_config(page_title="Healthcare Symptom Checker", page_icon="🏥", layout="centered")

st.title("🏥 Healthcare Symptom Checker — Educational Demo")
st.info("This tool is for educational purposes only. It does not provide medical advice.")

symptoms = st.text_area("Describe your symptoms:", placeholder="e.g. mild fever and sore throat")

if st.button("Check Symptoms"):
    if not symptoms.strip():
        st.warning("Please enter symptoms first.")
    else:
        st.info("Calling backend API...")
        resp = requests.post(API_LOCAL, json={"symptoms": symptoms})
        data = resp.json()
        st.success("Response received from backend ✅")
        st.subheader("🔎 Results")
        st.markdown(f"**Input:** {data['input']}")
        for cond in data["probable_conditions"]:
            st.markdown(f"**Condition:** {cond['condition']}  \nConfidence: {cond['confidence']}  \nRationale: {cond['rationale']}")
        st.subheader("🩺 Recommended Next Steps")
        for step in data["recommended_next_steps"]:
            st.write("•", step)
        st.caption(data["disclaimer"])

st.sidebar.header("📊 Query History (from DB)")
if os.path.exists(DB_PATH):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM queries ORDER BY id DESC LIMIT 10", conn)
    conn.close()
    st.sidebar.dataframe(df)
else:
    st.sidebar.info("No history yet.")
