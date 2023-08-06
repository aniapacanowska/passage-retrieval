'''
Retrieve the passages using Elasticsearch and calculate NDCG@10.
'''

import elastic
from utils import Solver

def get_top_N_elastic(index_name, path_in, N):
    results = elastic.top_k_matches(index_name, path_in, N)

    ids = []
    for question_id in results:
        passage_ids = []
        for passage_id, _ in results[question_id]:
            passage_ids.append(passage_id)
        ids.append(passage_ids)    
    
    return ids

index_name = 'passages-wiki-index'
path_in = 'data/test-B/in-wiki-trivia.tsv'
path_out = 'out/best/test-B-allegro.tsv'
path_expected = 'data/test-B/expected-allegro-faq.tsv'
ids = get_top_N_elastic(index_name, path_in, 10)
solver = Solver(None, None)
solver.save_passages(ids, path_out)
score = solver.score_passages(path_out, path_expected)
print(score)


# index_name = 'passages-wiki-freq-index'
# path_in = 'data/wiki-trivia/questions-train-freq.jl'
# path_out = 'data/lemmatization/freq-10000.tsv'
# ids = get_top_N_elastic(index_name, path_in, 10000)
# solver = Solver(None, None)
# solver.save_passages(ids, path_out)