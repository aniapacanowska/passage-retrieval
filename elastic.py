'''
Index the passages and retrieve top matches using Elasticsearch.
'''

from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm
import json


def create_passages_index(index_name):
    '''
    Creates a new index for passages. If one already exists, it is deleted.
    '''
    es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme':'http'}])
    settings = {
        'index': {
            'number_of_shards': 1,
            'number_of_replicas': 0,
        }
    }
    mappings = {
        'properties': {
            'id': {'type': 'text'},
            'text': {'type' : 'text'}
        }
    }

    if es.indices.exists(index=index_name):
        es.indices.delete(index = index_name)

    es.indices.create(
        index = index_name,
        settings = settings,
        mappings = mappings)
    es.transport.close()


def index_passages(path, index_name):
    es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme':'http'}])
    N=500
    passages = []
    for i, line in tqdm(enumerate(open(path))):
        passages.append(json.loads(line))
        if i%N == N-1:
            helpers.bulk(es, passages, index=index_name, request_timeout=30)
            passages = []
    helpers.bulk(es, passages, index=index_name, request_timeout=30)
    es.transport.close()


def top_k_matches(index_name, in_path, k):
    es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme':'http'}])
    es.indices.open(index=index_name)
    max_retries = 3
    results = {}
    for i, line in enumerate(tqdm(open(in_path))):
        if in_path[-3:] == '.jl':
            question_id = json.loads(line)['id']
            question_text = json.loads(line)['text']
        else:
            question_id = i
            question_text = ' '.join(line.split()[1:])

        query = {
            'match': {
                'text': question_text
            }
        }
        response = es.search(index=index_name, query=query, size=k)
        for i in range(max_retries):
            if response['hits']['total']['value'] < k:
                response = es.search(index=index_name, query=query, size=k)
            else:
                break
        if response['hits']['total']['value'] < k:
            print('Got only {0} passages for question {1}.'.format(
                response['hits']['total']['value'], question_text))

        scores = []
        for hit in response['hits']['hits']:
            scores.append((str(hit['_source']['id']), hit['_score']))
        results[question_id] = scores
    es.transport.close()
    return results



# index_name = 'passages-legal-index'
# path = 'data/legal-questions/passages.jl'

# create_passages_index(index_name)
# index_passages(path, index_name)

# top_k_matches('data/dev-mini/in-morfeusz.tsv', 'data/scores/mini.tsv', 10)