'''
Prepare datasets for BERT training and testing.
'''

import json
import random

def split_train_questions(train_path, dev_path, new_train_path, new_dev_path):
    '''
    Create new train and dev dataset from dataset at train_path.
    Question is in the new dev dataset if it was in the dataset at dev_path.
    This is because the original dev questions are a subset of the train questions. 
    '''
    train_f = open(train_path, 'r')
    dev_f = open(dev_path, 'r')
    dev_questions = set([' '.join(line.split()[1:]) for line in dev_f])
    train_questions = [json.loads(line) for line in train_f]

    new_train_f = open(new_train_path, 'w')
    new_dev_f = open(new_dev_path, 'w')
    for question in train_questions:
        if question['text'] in dev_questions:
            new_dev_f.write(json.dumps(question, ensure_ascii=False)+'\n')
        else:
            new_train_f.write(json.dumps(question, ensure_ascii=False)+'\n')


def first_k_ids(in_path, out_path, k):
    '''
    Get first k ids from each line in in_path and save them to out_path.
    '''
    out_ids = [' '.join(line.split()[:k]) for line in open(in_path)]
    f_out = open(out_path, 'w')
    f_out.write('\n'.join(out_ids))


def load_passages(path):
    '''
    Create dict id:passage for each passage at path.
    '''
    passages = {}
    for line in open(path):
        passage = json.loads(line)
        passages[str(passage['id'])] = passage['text']
    return passages


def load_questions(path):
    '''
    Create dict id:question for each question at path.
    '''
    questions = {}
    for line in open(path):
        question = json.loads(line)
        questions[question['id']] = question['text']
    return questions


def random_negatives(all_ids, positive_ids, k):
    '''
    Get k random negative ids (irrelevant passages) for one question. 
    '''
    negative_ids = []
    while len(negative_ids) < k:
        id = random.choice(all_ids)
        while id in negative_ids or id in positive_ids:
            id = random.choice(all_ids)
        negative_ids.append(id)
    return negative_ids


def get_positives(path, questions):
    '''
    Get dict id:list of relevant passages from pairs at path.
    '''
    positive_ids = {}
    for line in open(path).read().split('\n')[1:-1]:
        question_id = int(line.split()[0])
        passage_id = line.split()[1]

        if question_id not in questions:
            continue

        if question_id in positive_ids:
            positive_ids[question_id].append(passage_id)
        else:
            positive_ids[question_id] = [passage_id]
    return positive_ids


def get_negatives_random(all_ids, positive_ids, k):
    '''
    Get dict id:list of irrelevant passages.
    '''
    negative_ids = {}
    for question_id in positive_ids:
        negative_ids[question_id] = random_negatives(all_ids, positive_ids[question_id], k-len(positive_ids[question_id]))
    return negative_ids


def get_negatives_from_path(retrieved_path, questions, positive_ids, k):
    '''
    Get dict id:list of irrelevant passages.
    '''
    question_ids = sorted(questions.keys())
    negative_ids = {}
    for i, line in enumerate(open(retrieved_path)):
        retrieved_ids = line.split()
        question_id = question_ids[i]
        negative_ids[question_id] = []
        for retrieved_id in retrieved_ids:
            if retrieved_id not in positive_ids[question_id]:
                negative_ids[question_id].append(retrieved_id)
            if len(negative_ids[question_id])+len(positive_ids[question_id]) == k:
                break
    return negative_ids


def get_pairs(passages_path, questions_path, positives_path, retrieved_path, k, negatives_method='path'):
    '''
    Create a list of tuples (question_id, question_text, passage_id, passage_text, score).
    Score is 1 when passage is relevant, 0 otherwise.
    '''
    passages = load_passages(passages_path)
    questions = load_questions(questions_path)

    all_ids = list(passages.keys())
    positive_ids = get_positives(positives_path, questions)
    if negatives_method == 'random':
        negative_ids = get_negatives_random(all_ids, positive_ids, k)
    elif negatives_method == 'path':
        negative_ids = get_negatives_from_path(retrieved_path, questions, positive_ids, k)
    else:
        print('Unknown method of choosing negative passage_ids.')
        negative_ids = {}

    rows = []
    for question_id in positive_ids:
        for passage_id in positive_ids[question_id]:
            rows.append((question_id, questions[question_id], passage_id, passages[passage_id], 1))
    
    for question_id in negative_ids:
        for passage_id in negative_ids[question_id]:
            rows.append((question_id, questions[question_id], passage_id, passages[passage_id], 0))
    
    random.shuffle(rows)
    return rows


def get_test_pairs(passages_path, questions_path, pool_path):
    '''
    Create a list of tuples (question_text, passage_id, passage_text).
    passages_path: path to file with passage_id, passage_text
    questions_path: path to test file (CSV) with domain_name and question_text
    pool_path: path to file with passage_ids to be considered for re-ranking for each question
    '''
    passages = load_passages(passages_path)

    rows = []
    for line_question, line_pool in zip(open(questions_path), open(pool_path)):
        question_text = ' '.join(line_question.split()[1:])
        for passage_id in line_pool.split():
            rows.append((question_text, passage_id, passages[passage_id]))
    return rows


def save_test_pairs(rows, path):
    '''
    Save tuples of form (question_text, passage_id, passage_text) in the json format to the file at path.
    '''
    f = open(path, 'w')
    for row in rows:
        f.write(json.dumps({
            'question_text': row[0],
            'passages_id': row[1],
            'passage_text': row[2]
        }, ensure_ascii=False)+'\n')


def save_pairs(rows, path):
    '''
    Save tuples of form (question_id, question_text, passage_id, passage_text, label(0/1)) in the json format to the file at path.
    '''
    f = open(path, 'w')
    for row in rows:
        f.write(json.dumps({
            'question_id': row[0],
            'question_text': row[1],
            'passages_id': row[2],
            'passage_text': row[3],
            'label': row[4]
        }, ensure_ascii=False)+'\n')


# first_k_ids('data/bert/bm25-1000-morfeusz-dev.tsv', 'data/bert/bm25-10-morfeusz-dev.tsv', 10)
# rows = get_pairs('data/wiki-trivia/passages.jl', 'questions-train.jl', 'pairs-train.tsv', 'data/bert/bm25-1000-morfeusz-train.tsv', 100)
# save_pairs(rows, 'dataset-train-100-morfeusz.jl')
rows = get_test_pairs('data/wiki-trivia/passages.jl', 'data/dev-0/in-mini.tsv', 'data/bert/bm25-10-morfeusz-dev.tsv')
save_test_pairs(rows, 'dataset-test-10-dev-mini.jl')