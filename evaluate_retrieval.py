"""Evaluate fixed authored lexical-retrieval cases, not clinical effectiveness."""
import json
from pathlib import Path
from campuspulse.knowledge import CARDS, corpus_sha256, retrieve


def main():
    root=Path(__file__).resolve().parent
    cases=json.loads((root/'configs/retrieval_cases.json').read_text(encoding='utf-8'))
    rows=[]
    for case in cases:
        matches=retrieve(case['query'],limit=3)
        ids=[c['id'] for c in matches]
        ranks=[i+1 for i,ident in enumerate(ids) if ident in case['expected_ids']]
        rows.append({**case,'retrieved_ids':ids,'hit_at_1':bool(ranks and ranks[0]==1),
            'hit_at_3':bool(ranks),'reciprocal_rank_at_3':1/ranks[0] if ranks else 0})
    report={'evaluation_kind':'authored_fixed_query_regression_set','cards':len(CARDS),'queries':len(rows),
        'corpus_sha256':corpus_sha256(),'hit_at_1':sum(r['hit_at_1'] for r in rows)/len(rows),
        'hit_at_3':sum(r['hit_at_3'] for r in rows)/len(rows),
        'mrr_at_3':sum(r['reciprocal_rank_at_3'] for r in rows)/len(rows),'cases':rows,
        'limitations':['Queries and expected IDs were authored by the project builder; not independent or representative user evaluation.',
            'Labels assume one preferred card, although several cards may be relevant.',
            'Lexical overlap measures retrieval, not health benefit or semantic comprehension.',
            'No model call is required. New external queries and human judgments are still needed.']}
    out=root/'artifacts/retrieval';out.mkdir(parents=True,exist_ok=True)
    (out/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(f"Cards: {report['cards']} | Queries: {report['queries']}")
    print(f"Hit@1: {report['hit_at_1']:.0%} | Hit@3: {report['hit_at_3']:.0%} | MRR@3: {report['mrr_at_3']:.3f}")
    for r in rows:
        if not r['hit_at_1']:print('Top-1 miss:',r['query'],'->',r['retrieved_ids'])
    print('Saved: artifacts/retrieval/report.json (authored regression set, not an external benchmark)')

if __name__=='__main__':main()
