from transformers import pipeline

_NER_PIPE = None

def load_ner_model(model_name="tner/bertweet-base-twitterner2021"):
    global _NER_PIPE
    if _NER_PIPE is None:
        _NER_PIPE = pipeline(
            "ner",
            model=model_name,
            aggregation_strategy="simple",
            device=-1
        )
    return _NER_PIPE

BASE_SKILLS = [
    "python", "pandas", "numpy", "sql", "power bi", "excel",
    "machine learning", "deep learning", "pytorch", "tensorflow",
    "docker", "kubernetes", "aws", "react", "flask"
]

def extract_skills_from_ner(text, model_name="tner/bertweet-base-twitterner2021"):
    try:
        ner = load_ner_model(model_name)
        entities = ner(text)

        found = set([e['word'].lower().strip() for e in entities if len(e['word']) > 2])

        
        text_l = text.lower()
        for kw in BASE_SKILLS:
            if kw in text_l:
                found.add(kw)

        return sorted(found)
    except:
        return []