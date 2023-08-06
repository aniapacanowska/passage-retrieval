import json
import math
import numpy as np

class Dataset:
    def __init__(self, path, max_len=100000, use_arrays=False):
        '''
        path: path to the json file with passages
        max_len: maximum length of the passage. If a passage is loneger, it is split into smaller pieces
        use_arrays: if True, use numpy arrays for ids and passages (this reduces memory usage). Otherwise use lists.
        '''
        self.max_len = max_len
        self.use_arrays = use_arrays
        self.passages = []
        self.ids = []
        self.load_passages(path)

    def tokenize(self, text):
        '''
        Create a list of tokens from text:
        Split the text by ' ' and '\n', remove all non-alphanumeric characters and lowercase all characters.
        '''
        tokens = []
        for token in text.replace('\n', ' ').split():
            token = ''.join(filter(str.isalnum, token)).lower()
            if token:
                tokens.append(token)

        if self.use_arrays:
            tokens = np.array(tokens)
        return tokens

    def load_passages(self, path):
        '''
        Load passages from file at path and tokenize them.
        '''
        for line in open(path):
            passage = json.loads(line)
            tokens = self.tokenize(passage['text'])
            for i in range(0, len(tokens), self.max_len):
                self.passages.append(tokens[i:min(i+self.max_len, len(tokens))])
                self.ids.append(str(passage['id']))

        if self.use_arrays:
            self.passages = np.array(self.passages, dtype=object)
            self.ids = np.array(self.ids)



class Solver:
    def __init__(self, dataset, model):
        self.dataset = dataset
        self.model = model

    def find_passages(self, in_path):
        '''
        Find ids of the matching passages for each question in the file at in_path. 
        '''
        matching_ids = []
        questions = []
        for line in open(in_path).read().split('\n')[:-1]:
            questions.append(self.dataset.tokenize(' '.join(line.split()[1:])))
        
        matching_ids = self.model.top_k_matches(np.array(questions, dtype=object), 10)
        return matching_ids

    def save_passages(self, ids, out_path):
        '''
        Save the ids at out_path.
        '''
        out_file = open(out_path, 'w')
        for passage_ids in ids:
            out_file.write('\t'.join(passage_ids)+'\n')

    def score_passages(self, predicted_path, expected_path):
        '''
        Calculate the average of 10 Normalized Discounted Cumulative Gain (NDCG@10) for each question.
        '''
        score = 0
        for predicted_line, expected_line in zip(open(predicted_path).read().split('\n')[:-1], open(expected_path).read().split('\n')[:-1]):
            predicted_ids = predicted_line.split()
            expected_ids = expected_line.split()

            dcg = 0
            for i, id in enumerate(predicted_ids):
                if id in expected_ids:
                    dcg += 1/math.log2(i+2)
            idcg = sum([1/math.log2(i+2) for i in range(len(expected_ids))])

            score += dcg/idcg
        return 100*score/(len(open(predicted_path).read().split('\n'))-1)