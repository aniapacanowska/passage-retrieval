'''
Calculate CG, DCG, NDCG.
'''

from math import log2

def cg(rel_scores, ranking, N):
    score = 0
    for passage in ranking[:N]:
        score += rel_scores[passage]
    return score

def dcg(rel_scores, ranking, N):
    score = 0
    for i, passage in enumerate(ranking[:N]):
        score += rel_scores[passage]/log2(i+2)
    return score

def idcg(rel_scores, N):
    score = 0
    best_ranking = [k for k, v in sorted(rel_scores.items(), key=lambda x: -x[1])]
    for i, passage in enumerate(best_ranking[:N]):
        score += rel_scores[passage]/log2(i+2)
    return score

def ndcg(rel_scores, ranking, N):
    return dcg(rel_scores, ranking, N)/idcg(rel_scores, N)

def all_scores(rel_scores, ranking, N):
    print(cg(rel_scores, ranking, N))
    print(dcg(rel_scores, ranking, N))
    print(idcg(rel_scores, N))
    print(ndcg(rel_scores, ranking, N))

rel_scores = {"p1": 0.5, "p2": 0.0, "p3": 1.0, "p4": 0.3}
ranking = ["p3", "p4", "p1", "p2"]
N = 2
all_scores(rel_scores, ranking, N)