'''
Calculate BM25 scores and retrieve passages.
'''

import json
import math
import numpy as np
from tqdm import tqdm

from utils import Dataset, Solver

class BM25:
    def __init__(self, passages, ids, use_arrays=False):
        '''
        Model calculating BM25 scores.
        passeges: list of lists of tokens (or array of array of tokens)
        ids: list (or array) of ids matching passages
        use_arrays: when True, use numpy arrays. When False, use lists.
            Using numpy arrays reduces memory usage and is necessary to fit 
            all wikipedia passages into 32GB RAM. However, it makes the calculations run about 2x longer.
        '''
        self.use_arrays = use_arrays
        self.k1 = 1.2
        self.b = 0.75
        self.passages = passages
        self.ids = ids
        self.calculate_statistics()

    def calculate_statistics(self): 
        '''
        Calculate statistics necessary for BM25: Inverted Document Frequency for all terms and average document length.
        '''
        term_docs_count = {}
        for passage in self.passages:
            for term in set(passage):
                if term in term_docs_count:
                    term_docs_count[term] += 1
                else:
                    term_docs_count[term] = 1

        self.N = len(self.passages)
        self.idfs = {}
        for term in term_docs_count:
            n = term_docs_count[term]
            self.idfs[term] = math.log((self.N-n+0.5)/(n+0.5) + 1)

        doc_lens = [len(passage) for passage in self.passages]
        self.avgdl = sum(doc_lens)/len(doc_lens)

    def get_score(self, question, passage):
        '''
        Calculate BM25 score for given question and passage.
        '''
        if len(passage) == 0:
                return 0

        if self.use_arrays:
            equal_terms = (passage == question[:, None])
            if np.count_nonzero(equal_terms) == 0:
                return 0

        score = 0
        for i, term in enumerate(question):
            if self.use_arrays:
                tf = np.count_nonzero(equal_terms[i])
            else:
                tf = passage.count(term)

            if tf > 0:
                score += self.idfs[term]*tf*(self.k1+1)/(tf+self.k1*(1-self.b+self.b*len(passage)/self.avgdl))
        return score

    def top_k_matches(self, questions, k=10):
        '''
        Retrieve k best matching passages for each question in questions.
        questions: list of lists (or array of arrays) of tokens
        '''
        top_matches = []
        for question in tqdm(questions):
            scores = [(self.get_score(question, passage), i) for i, passage in enumerate(self.passages)]
            ids = set()
            top_idxs = []
            for s,idx in sorted(scores, reverse=True):
                id = self.ids[idx]
                if id not in ids: # make sure we get unique indexes (if passage is long, one id can occur more then once)
                    top_idxs.append(idx)
                    ids.add(id)
                if len(top_idxs) == k:
                    break
            top_matches.append([self.ids[idx] for idx in top_idxs])
        return top_matches


# solver = Solver(None, None)
# print(solver.score_passages('out/BERT_reranking(100-morfeusz)_10_morfeusz_1epoch/out-dev-morfeusz-10-1epoch.tsv', 'data/dev-0/expected.tsv'))

# dataset = Dataset('data/large/passages.jl')
# model = BM25(dataset.passages, dataset.ids)
# solver = Solver(dataset, model)
# passages = solver.find_passages('data/dev-small/in.tsv')


# solver.save_passages(passages, 'out.tsv')
# passages = [row.split() for row in open('out_bm25_dev.tsv').read().split('\n')[:-1]]
# print('NDCG:', solver.score_passages('out.tsv', 'data/dev-small/expected.tsv'))