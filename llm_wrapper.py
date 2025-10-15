%%writefile /content/health-symptom-checker/src/llm_wrapper.py
"""
Clean, self-contained llm_wrapper for the Symptom Checker project.

Provides:
- call_openai_llm: calls OpenAI (if key present) or mock; logs raw outputs
- parse_and_validate_json: robust JSON extraction + rescue for missing relative_score
- log_query: writes an anonymized entry to history.db
- get_safely_inferred: orchestrates rule-based -> emergency -> llm fallback
"""

import os
import json
import re
import sqlite3
import hashlib
from datetime import datetime
from typing import List, Tuple

# Detect whether to use OpenAI (set OPENAI_API_KEY in env to enable)
USE_OPENAI = bool(os.environ.get("OPENAI_API_KEY"))
try:
    import openai
except Exception:
    openai = None
    USE_OPENAI = False

# local imports (project)
from pydantic_models import SymptomResponse, Condition
from rule_based_v2 import infer_conditions

DB_PATH = "/content/health-symptom-checker/history.db"
RAW_LOG = "/content/health-symptom-checker/llm_raw_logs.txt"

# Simple rate limiter placeholder (always allow here)
class _AllowAll:
    def allow(self):
        return True
LLM_RATE_LIMITER = _AllowAll()

# Mock LLM: returns well-formed JSON string complying with SymptomResponse schema
def mock_llm(symptoms: str) -> str:
    out = {
        "input": symptoms,
        "probable_conditions": [
            {"condition": "Unclear — further questions required", "rationale": "Based on matching keywords.", "confidence": "low", "relative_score": 0.0}
        ],
        "recommended_next_steps": [
            "Educational only. Monitor symptoms and consult a healthcare provider if worsening.",
            "If severe or emergency signs: seek emergency care."
        ],
        "disclaimer": "Educational use only. Not medical advice."
    }
    return json.dumps(out, ensure_ascii=False)

def _ensure_log_dir():
    d = os.path.dirname(RAW_LOG)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def call_openai_llm(symptoms: str, timeout_secs: int = 15) -> str:
    """
    Returns raw string (LLM output) — either from OpenAI or the mock.
    Always appends the raw output to RAW_LOG for debugging.
    """
    _ensure_log_dir()
    # If OpenAI is not configured, use mock
    if not USE_OPENAI or openai is None:
        raw = mock_llm(symptoms)
        try:
            with open(RAW_LOG, "a", encoding="utf-8") as f:
                f.write("----MOCK CALL----\n")
                f.write(raw + "\n")
        except Exception:
            pass
        return raw

    # Use OpenAI (handle exceptions)
    if not LLM_RATE_LIMITER.allow():
        raise RuntimeError("LLM rate limit exceeded")

    system_msg = "You are a conservative educational medical assistant. Output ONLY valid JSON with no extra text."
    # PROMPT_TEMPLATE: use a literal with {symptoms} placeholder which we'll replace (not .format)
    PROMPT_TEMPLATE = """
You are a conservative educational medical assistant. IMPORTANT: Output ONLY valid JSON (no commentary, no code fences, no explanation). If you cannot produce valid JSON, output {"error":"cannot_respond"}.

Output must match this schema exactly:

{
  "input": "<original symptoms string>",
  "probable_conditions": [
    {"condition":"", "rationale":"", "confidence":"", "relative_score": 0.0}
  ],
  "recommended_next_steps": ["..."],
  "disclaimer": "..."
}

Rules:
- Provide 1-5 probable_conditions with short rationale (1-2 sentences) and confidence (low/medium/high).
- If emergency signs exist (chest pain, severe breathlessness, severe bleeding, fainting) return a single high-confidence emergency condition and recommended_next_steps starting with "Seek emergency care immediately".
- Do NOT include treatments, dosages, or prescriptions.
Input symptoms: {symptoms}
"""
    user_msg = PROMPT_TEMPLATE.replace("{symptoms}", symptoms)
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"system","content":system_msg},{"role":"user","content":user_msg}],
            temperature=0.0,
            max_tokens=700,
            request_timeout=timeout_secs
        )
        text = resp["choices"][0]["message"]["content"]
    except Exception as e:
        # on any error, fall back to mock and log the exception
        try:
            with open(RAW_LOG, "a", encoding="utf-8") as f:
                f.write("----OPENAI_ERROR----\n")
                f.write(str(e) + "\n")
        except Exception:
            pass
        text = mock_llm(symptoms)
        try:
            with open(RAW_LOG, "a", encoding="utf-8") as f:
                f.write("----FALLBACK_MOCK_OUTPUT----\n")
                f.write(text + "\n")
        except Exception:
            pass
    else:
        # log successful raw output
        try:
            with open(RAW_LOG, "a", encoding="utf-8") as f:
                f.write("----CALL----\n")
                f.write(text + "\n")
        except Exception:
            pass
    return text

def parse_and_validate_json(raw_text: str) -> SymptomResponse:
    """
    Extract JSON from raw_text and ensure fields required by SymptomResponse exist.
    Adds `relative_score` to probable_conditions entries if missing (derived from confidence).
    """
    raw = raw_text.strip()
    # strip triple-backtick fences if present
    if raw.startswith("```") and raw.endswith("```"):
        raw = "\n".join([l for l in raw.splitlines() if not l.strip().startswith("```")]).strip()

    # find largest {...} block
    candidates = []
    for start_ch, end_ch in [("{","}"), ("[","]")]:
        for m in re.finditer(re.escape(start_ch), raw):
            si = m.start()
            depth = 0
            for j in range(si, len(raw)):
                if raw[j] == start_ch:
                    depth += 1
                elif raw[j] == end_ch:
                    depth -= 1
                    if depth == 0:
                        candidates.append(raw[si:j+1])
                        break
    candidate = max(candidates, key=len) if candidates else None
    if candidate is None:
        fi = raw.find("{"); la = raw.rfind("}")
        if fi != -1 and la != -1 and la > fi:
            candidate = raw[fi:la+1]
    if candidate is None:
        raise ValueError("Could not locate JSON in LLM output")

    # Try several JSON fixes
    def try_load(s):
        try:
            return json.loads(s)
        except Exception:
            s2 = re.sub(r",\s*([}\]])", r"\1", s)  # remove trailing commas
            try:
                return json.loads(s2)
            except Exception:
                s3 = s2.replace("'", '"')
                return json.loads(s3)

    parsed = try_load(candidate)

    # rescue missing relative_score
    pcs = parsed.get("probable_conditions", [])
    rescued = False
    for pc in pcs:
        if "relative_score" not in pc:
            conf = str(pc.get("confidence", "")).lower()
            if conf == "high":
                pc["relative_score"] = 0.75
            elif conf == "medium":
                pc["relative_score"] = 0.5
            elif conf == "low":
                pc["relative_score"] = 0.25
            else:
                pc["relative_score"] = 0.33
            rescued = True

    if rescued:
        parsed.setdefault("notes", "")
        parsed["notes"] = (parsed.get("notes","") + " parsed_and_rescued_relative_score").strip()

    # Validate with Pydantic
    resp = SymptomResponse(**parsed)
    return resp

def _init_db():
    # idempotent create
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS queries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symptom_hash TEXT,
        timestamp_utc TEXT,
        engine TEXT,
        top_condition TEXT,
        top_score REAL,
        notes TEXT
    );
    """)
    conn.commit()
    conn.close()

def log_query(symptom_text: str, engine: str, top_condition: str, top_score: float, notes: str = ""):
    _init_db()
    h = hashlib.sha256(symptom_text.strip().lower().encode("utf-8")).hexdigest()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT INTO queries (symptom_hash, timestamp_utc, engine, top_condition, top_score, notes) VALUES (?, datetime('now'), ?, ?, ?, ?)",
                (h, engine, top_condition, float(top_score), notes))
    conn.commit()
    conn.close()

def get_safely_inferred(symptoms: str, allow_llm: bool = True, fuzzy_label_choices: List[str] = None) -> SymptomResponse:
    """
    Primary orchestration:
    - emergency short-circuit
    - rule-based inference
    - if none/confident, return
    - else fallback to LLM (if allowed)
    """
    # Emergency keywords (simple)
    s = symptoms.lower()
    emergencies = ["chest pain", "severe chest pain", "difficulty breathing", "shortness of breath", "unconscious", "severe bleeding", "fainting"]
    for e in emergencies:
        if e in s:
            # return emergency response
            out = {
                "input": symptoms,
                "probable_conditions": [{"condition":"Possible emergency — seek immediate care","rationale":"Emergency keyword matched.","confidence":"high","relative_score":1.0}],
                "recommended_next_steps":["Seek emergency care immediately."],
                "disclaimer":"Educational only. Not medical advice."
            }
            log_query(symptoms, "emergency", out["probable_conditions"][0]["condition"], 1.0, "emergency_short_circuit")
            return SymptomResponse(**out)

    # Rule-based
    rules = infer_conditions(symptoms)  # list of (cond, score)
    if rules and rules[0][1] > 0.0:
        pcs = [{"condition":cond, "rationale":"Matched rule keywords.","confidence":"medium" if score<1.0 else "high", "relative_score":score} for cond,score in rules]
        out = {"input": symptoms, "probable_conditions": pcs, "recommended_next_steps":["Educational only. Consider seeing a clinician for evaluation."], "disclaimer":"Educational only. Not medical advice."}
        # log top rule
        log_query(symptoms, "rule_based", pcs[0]["condition"], pcs[0]["relative_score"], "rule_based_match")
        return SymptomResponse(**out)

    # Fallback to LLM if allowed
    if allow_llm:
        raw = call_openai_llm(symptoms)
        try:
            validated = parse_and_validate_json(raw)
            # log success
            top = validated.probable_conditions[0]
            log_query(symptoms, "llm", top.condition, getattr(top, "relative_score", 0.0), getattr(validated, "notes", ""))
            return validated
        except Exception as e:
            # log parse error and fallback response
            notes = f"llm_parse_error:{str(e)}"
            log_query(symptoms, "fallback", "Unclear — further questions required", 0.0, notes)
            fallback = {
                "input": symptoms,
                "probable_conditions": [{"condition":"Unclear — further questions required","rationale":"Fallback due to LLM parse/validation error.","confidence":"low","relative_score":0.0}],
                "recommended_next_steps":["Educational only. Consider seeing a clinician for evaluation."],
                "disclaimer":"Educational only. Not medical advice."
            }
            return SymptomResponse(**fallback)
    else:
        # LLM not allowed: safe fallback
        log_query(symptoms, "no_llm", "Unclear — further questions required", 0.0, "llm_disabled")
        fallback = {
            "input": symptoms,
            "probable_conditions": [{"condition":"Unclear — further questions required","rationale":"LLM disabled or no rules matched.","confidence":"low","relative_score":0.0}],
            "recommended_next_steps":["Educational only. Consider seeing a clinician for evaluation."],
            "disclaimer":"Educational only. Not medical advice."
        }
        return SymptomResponse(**fallback)
