'''
Retrieve the passages using word2vec model.
'''

import gensim
from gensim.models import Word2Vec
from scipy.spatial import distance
import numpy as np

from utils import Dataset, Solver

class Word2VecModel:
    def __init__(self, passages, ids, testing=False, vector_size=100):
        '''
        Class for calculating scores based on word2vec.
        '''
        path = 'models/nkjp+wiki-forms-all-100-skipg-ns.txt'
        self.vector_size = vector_size
        if testing: # avoid loading large model for testing
            self.wv = Word2Vec(sentences = [['Tu', 'stoi', 'piękny', 'buk'], 'Tam', 'buk', 'czerwony'], min_count=1).wv
        else:
            self.wv = gensim.models.KeyedVectors.load_word2vec_format(path, binary=False)
        self.passages = passages
        self.ids = ids
        self.passage_vectors = []
        self.compute_passage_vectors()

    def get_document_vector(self, doc):
        '''
        Calculate vector representation of a document.
        '''
        term_vectors = [self.wv[term] for term in doc if term in self.wv]
        if len(term_vectors) == 0:
            return np.ones(self.vector_size)
        return np.average(term_vectors, axis=0)

    def compute_passage_vectors(self): 
        '''
        Calculate vector representation for each passage.
        '''
        self.passage_vectors = []
        for passage in self.passages:
            self.passage_vectors.append(self.get_document_vector(passage))

    def get_scores(self, question_vectors, passage_vectors):
        '''
        Calculate similarity score for given vectors of question and passage.
        '''
        return np.dot(question_vectors, np.array(passage_vectors).T)/(
            np.linalg.norm(question_vectors, axis=1)[:,None]*np.linalg.norm(passage_vectors, axis=1))

    def top_k_matches(self, questions, k=10):
        '''
        Retrieve k passages best matching given question.
        '''
        question_vectors = np.array([self.get_document_vector(question) for question in questions])
        scores = self.get_scores(question_vectors, self.passage_vectors)
        top_ids = np.argpartition(-scores, k-1)[:, :k] # use parition to avoid sorting entire array
        top_scores = np.take_along_axis(scores, top_ids, 1)
        sorted_top_ids = np.take_along_axis(top_ids, np.argsort(-top_scores), 1)
        return [[self.ids[id] for id in row] for row in sorted_top_ids]

class Word2VecModelTest:
    def __init__(self):
        '''
        Class for simple test of Word2VecModel.
        '''
        self.passages = [
            ['pies', 'kot', 'tygrys', 'żółw'],
            ['słoń', 'żyrafa', 'pies'], 
            ['kawka', 'żółw', 'sójka', 'tygrys']]
        self.ids = [4,5,6]
        self.questions = [
            ['gdzie', 'jest', 'pies'],
            ['czy', 'w', 'zoo', 'są', 'tygrys', 'i', 'żyrafa']
        ]
        self.model = Word2VecModel(self.passages, self.ids, testing = True, vector_size = 3)
        self.model.wv = {
            'pies': np.array([1, 2, 3]),
            'kot': np.array([-2, 2, 4]),
            'tygrys': np.array([-2, 2, 5]),
            'żółw': np.array([-4, 0, -4]),
            'żyrafa': np.array([3, -3, 3]),
            'kawka': np.array([0, 4, 3]),
            'sójka': np.array([1, 7, 0])}
        self.model.compute_passage_vectors()

    def test(self):
        assert((self.model.get_document_vector(['tygrys']) == [-2, 2, 5]).all())
        assert((self.model.get_document_vector(self.passages[0]) == [-1.75, 1.5, 2]).all())
        assert((self.model.get_document_vector(self.passages[1]) == [2, -0.5, 3]).all())

        question_vectors = np.array([self.model.get_document_vector(question) for question in self.questions])
        assert((question_vectors[0] == [1, 2, 3]).all())
        assert((question_vectors[1] == [0.5, -0.5, 4]).all())

        scores = self.model.get_scores(question_vectors, self.model.passage_vectors)
        assert (np.allclose(scores, [[0.63495193, 0.7342231, 0.60861167], [0.51428644, 0.89611958, 0.11891768]]))

        assert (self.model.top_k_matches(self.questions, k=2) == [[5,4], [5,4]])
        assert (self.model.top_k_matches(self.questions, k=3) == [[5,4,6], [5,4,6]])


test = Word2VecModelTest()
test.test()

dataset = Dataset('data/large/passages.jl', use_arrays=True)
model = Word2VecModel(dataset.passages, dataset.ids)
solver = Solver(dataset, model)

path_out = 'out.tsv'
passages = solver.find_passages('data/dev-small/in.tsv')
solver.save_passages(passages, path_out)
print('NDCG:', solver.score_passages(path_out, 'data/dev-small/expected.tsv'))