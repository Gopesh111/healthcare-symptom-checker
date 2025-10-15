import re

SYNONYMS = {
    "throwing up": "vomiting",
    "throw up": "vomiting",
    "belly ache": "abdominal pain",
    "stomach ache": "abdominal pain",
    "feverish": "fever",
    "breathless": "shortness of breath",
    "light headed": "lightheaded",
    "soar throat": "sore throat",
    "soar": "sore",
    "feaver": "fever",
    "temprature": "temperature"
}

RULES = {
    "fever": ["flu", "infection"],
    "sore throat": ["throat infection", "cold"],
    "vomiting": ["food poisoning", "gastroenteritis"],
    "cough": ["cold", "bronchitis"],
    "headache": ["migraine", "stress"],
    "fatigue": ["anemia", "stress", "sleep deprivation"]
}

def normalize_text(text):
    text = text.lower()
    for k, v in SYNONYMS.items():
        text = text.replace(k, v)
    text = re.sub(r"[^a-z\s]", " ", text)
    return text.strip()

def infer_conditions(symptom_text):
    text = normalize_text(symptom_text)
    results = []
    for key, conditions in RULES.items():
        if key in text:
            for cond in conditions:
                results.append((cond, 1.0))
    return results or []
