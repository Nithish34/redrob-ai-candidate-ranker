import json
from scorer import score_from_json_bytes
from pathlib import Path

raw = Path(__file__).with_name('sample_candidates.json').read_bytes()
results = score_from_json_bytes(raw)
print(f'Total candidates scored: {len(results)}')
print()
print('Top 5:')
for i, r in enumerate(results[:5], 1):
    cid = r['candidate_id']
    sc = r['score']
    rs = r['reasoning']
    print(f'  {i}. {cid}  score={sc:.4f}  {rs}')

print()
print('Bottom 3:')
for i, r in enumerate(results[-3:], len(results)-2):
    cid = r['candidate_id']
    sc = r['score']
    rs = r['reasoning']
    print(f'  {i}. {cid}  score={sc:.4f}  {rs}')

# Check non-increasing
scores = [r['score'] for r in results]
ok = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
print(f'\nScores non-increasing: {ok}')

# Show component breakdown of #1
print('\nComponent breakdown for #1:')
for k, v in results[0]['components'].items():
    print(f'  {k}: {v}')
