"""Local curated lexical retrieval; no browsing or model call during search."""
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

PATH = Path(__file__).resolve().parents[1]/'configs'/'support_cards.json'
CARDS = json.loads(PATH.read_text(encoding='utf-8'))
if len({c['id'] for c in CARDS}) != len(CARDS):
    raise ValueError('Support card IDs must be unique.')
BY_ID = {c['id']:c for c in CARDS}
CORPUS = [f"{c['title']} {c['text']} {c['tags']}" for c in CARDS]
VECTORIZER = TfidfVectorizer(stop_words='english',ngram_range=(1,2),sublinear_tf=True).fit(CORPUS)
MATRIX = VECTORIZER.transform(CORPUS)


def retrieve(query, topic=None, limit=2):
    if not isinstance(query,str) or len(query)>240:
        raise ValueError('Support query must be text of at most 240 characters.')
    scores = (MATRIX @ VECTORIZER.transform([query]).T).toarray().ravel()
    eligible = [i for i,c in enumerate(CARDS) if topic is None or c['topic']==topic]
    ranked = sorted(eligible,key=lambda i:(-scores[i],i))[:limit]
    # A zero lexical match is not evidence of relevance.
    return [{**CARDS[i],'similarity':round(float(scores[i]),4)} for i in ranked if scores[i]>0]


def approved_wordings(card):
    result = [card['text']]
    if card['topic']=='movement':
        tail=card['text'].removeprefix('If comfortable and accessible, ')
        result.append('If you would like, and it feels comfortable and accessible, '+tail)
    elif card['text'].startswith('If it suits you, '):
        result.append(card['text'].replace('If it suits you, ','If you would like, ',1))
    elif card['text'].startswith('If you want, '):
        result.append(card['text'].replace('If you want, ','If you would like, ',1))
    return result


def corpus_sha256():
    return hashlib.sha256(PATH.read_bytes()).hexdigest()
