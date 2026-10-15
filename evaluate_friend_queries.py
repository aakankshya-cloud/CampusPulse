"""Replay exposed friend queries; after tuning this is a regression set."""
import hashlib
import json
from pathlib import Path
from campuspulse.knowledge import retrieve, unsupported_request, corpus_sha256


def main():
    root=Path(__file__).resolve().parent
    source=root/'artifacts/retrieval/friend_queries_before.json'
    before=json.loads(source.read_text(encoding='utf-8'))
    rows=[]
    for row in before['cases']:
        matches=retrieve(row['query'],limit=3)
        refusal=unsupported_request(row['query'])
        rows.append({**row,'current_top3':[c['id'] for c in matches],
                     'scope_message':refusal,'top_text':matches[0]['text'] if matches else None})
        print(row['id'], '->', refusal or ([c['id'] for c in matches] or 'no_match'))
    report={'evaluation_kind':'friend_authored_queries_reused_for_development',
            'provenance':before['provenance'], 'corpus_sha256':corpus_sha256(),
            'retriever_sha256':hashlib.sha256((root/'campuspulse/knowledge.py').read_bytes()).hexdigest(),
            'limitations':['Queries were inspected before changes: post-change results are regression evidence, not held-out performance.',
                           'Scope rules are lexical and incomplete; general intent understanding is not demonstrated.',
                           'No independent relevance labels or accuracy metric; no live LLM calls.'], 'cases':rows}
    out=root/'artifacts/retrieval/friend_queries_after.json'
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Saved:',out)


if __name__=='__main__':
    main()
