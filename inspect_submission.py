import csv

rows = list(csv.DictReader(open('submission.csv', encoding='utf-8')))
print(f'Total rows: {len(rows)}')
print()
print('Top 10:')
for r in rows[:10]:
    rk = r['rank']
    cid = r['candidate_id']
    sc = r['score']
    rs = r['reasoning']
    print(f'  #{rk:>3}  {cid}  score={sc}  {rs}')

print()
print('Rank 50 and 100:')
r50 = rows[49]
r100 = rows[99]
print(f'  #{r50["rank"]:>3}  {r50["candidate_id"]}  score={r50["score"]}  {r50["reasoning"]}')
print(f'  #{r100["rank"]:>3}  {r100["candidate_id"]}  score={r100["score"]}  {r100["reasoning"]}')

scores = [float(r['score']) for r in rows]
print()
print(f'Score range    : {scores[-1]:.4f} -- {scores[0]:.4f}')
ok = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
print(f'Non-increasing : {ok}')
ids_unique = len(set(r['candidate_id'] for r in rows)) == 100
print(f'Unique IDs     : {ids_unique}')
ranks_ok = sorted([int(r['rank']) for r in rows]) == list(range(1, 101))
print(f'Ranks 1-100    : {ranks_ok}')
