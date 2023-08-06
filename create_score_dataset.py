'''
Create a dataset with question, passage pairs and BM25-based scores.
'''

import elastic
import json

from bm25_utils import BM25

def get_questions(path_questions):
    questions = {}
    for i, line in enumerate(open(path_questions)):
        if path_questions[-3:] == '.jl':
            row = json.loads(line)
            questions[row['id']] = row['text']
        else:
            questions[i] = ' '.join(line.split()[1:])
    return questions


def get_passages(ids, path_passages):
    passages = {}
    for line in open(path_passages):
        row = json.loads(line)
        id = str(row['id'])
        if id in ids:
            passages[id] = row['text']
    return passages


def write_json_format(path_out, rows):
    f_out = open(path_out, 'w')
    for row in rows:
        f_out.write(json.dumps(row, ensure_ascii=False)+'\n')


def get_top_N_bm25(index_name, path_questions, path_passages, path_in,  N): # path_in needs the same lemmatization as index_name, path_questions is unlemmatized
    results = elastic.top_k_matches(index_name, path_in, N)
    questions = get_questions(path_questions)

    passage_ids = set()
    for question_id in results:
        for passage_id, _ in results[question_id]:
            passage_ids.add(passage_id)
    passages = get_passages(passage_ids, path_passages)
    
    rows = []
    for question_id in results:
        scores = results[question_id]
        for passage_id, score in scores:
            rows.append({
                'question_id': question_id,
                'question_text': questions[question_id],
                'passage_id': passage_id,
                'passage_text': passages[passage_id],
                'score_bm25': score
            })
    return rows


def read_rows(path):
    rows = []
    for line in open(path):
        rows.append(json.loads(line))
    return rows


def add_bm25_scores(rows, use_bigrams, path_stats):
    bm25 = BM25(use_bigrams)
    if use_bigrams:
        score_name = "score_bm25_bigrams"
    else:
        score_name = "score_bm25_not_lemmatized"

    bm25.load_statistics(path_stats)
    for row in rows:
        row[score_name] = bm25.get_score(row["question_text"], row["passage_text"])
    return rows
    

path_dataset = 'test-B/big/allegro.jl'
path_terms = 'data/bm25/allegro-terms.jl'
path_bigrams = 'data/bm25/allegro-bigrams.jl'


path_questions = 'test-B/in/allegro.tsv'
path_in = 'test-B/in/allegro-morfeusz.tsv'
path_passages = 'data/allegro-faq/passages.jl'
rows = get_top_N_bm25('passages-allegro-morfeusz-index', path_questions, path_passages, path_in, 1000)
rows = add_bm25_scores(rows, False, path_terms)
rows = add_bm25_scores(rows, True, path_bigrams)
write_json_format(path_dataset, rows)
