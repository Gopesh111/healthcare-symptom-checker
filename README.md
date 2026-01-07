# Healthcare Symptom Checker 

###  Overview
This project is a simple LLM-powered Flask + Streamlit app that:
- Accepts symptom text input  
- Suggests **possible conditions** using a rule-based and LLM-fallback system  
- Displays **educational recommendations** (not medical advice)

---

### Tech Stack
- **Flask (Backend API)**
- **Streamlit (Frontend UI)**
- **Pydantic (Schema validation)**
- **SQLite (Query history)**

---

###  Workflow
1. User enters symptoms  
2. Backend checks:
   - **Emergency keywords**
   - **Rule-based matching**
   - **LLM fallback (mocked if API key missing)**
3. Returns structured JSON to UI

---

### Run Locally (Colab or Desktop)
```bash
pip install -r requirements.txt
python src/app.py
