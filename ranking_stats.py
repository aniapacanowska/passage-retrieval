'''
Count the number of relevant passages in different fragments of the ranking,
'''

import json

ids = []
for line in open('data/bert/questions-train.jl'):
    id = json.loads(line)['id']
    ids.append(id)

passages = {}
for i, line in enumerate(open('data/lemmatization/freq-10000.tsv')):
    passage_ids = line.split()
    passages[ids[i]] = passage_ids

good = 0
semi0 = 0
semi1 = 0
semi2 = 0
bad = 0
for line in open('data/wiki-trivia/pairs-train.tsv'):
    try:
        question_id = int(line.split()[0])
    except:
        continue
    passage_id = line.split()[1]
    if question_id in passages:
        if passage_id in passages[question_id][:10]:
            good += 1
        elif passage_id in passages[question_id][:100]:
            semi0 += 1
        elif passage_id in passages[question_id][:1000]:
            semi1 += 1
        elif passage_id in passages[question_id]:
            semi2 += 1
        else:
            bad +=1

print(good, semi0, semi1, semi2, bad)