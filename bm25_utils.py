'''
BM25 to use in creating the scores dataset.
'''


import json
import math
import numpy as np
from tqdm import tqdm

class BM25:
    def __init__(self, use_bigrams):
        self.k1 = 1.2
        self.b = 0.75
        self.use_bigrams = use_bigrams


    def get_tokens_len(self, tokens):
        passage_len = len(tokens)
        if self.use_bigrams:
            passage_len -= 1
        return passage_len

    
    def get_term(self, tokens, i):
        term = tokens[i]
        if self.use_bigrams:
            term = ' '.join([tokens[i], tokens[i+1]])
        return term


    def get_all_terms(self, tokens):
        if self.use_bigrams:
            return [' '.join([tokens[i], tokens[i+1]]) for i in range(len(tokens)-1)]
        else:
            return tokens


    def tokenize(self, text):
        tokens = []
        for token in text.split():
            token = ''.join(filter(str.isalnum, token)).lower()
            if token:
                tokens.append(token)
        return tokens


    def load_statistics(self, path):
        f = open(path)
        self.N = json.loads(f.readline())["docs_number"]
        self.avgdl = json.loads(f.readline())["average_length"]
        self.idfs = {}
        for line in f:
            row = json.loads(line)
            self.idfs[row["term"]] = row["idf"]


    def calculate_statistics(self, passages_path): 
        term_docs_count = {}
        doc_lens = []
        self.N = 0
        for line in open(passages_path):
            passage_text = self.tokenize(json.loads(line)['text'])
            passage_len = self.get_tokens_len(passage_text)
            doc_lens.append(len(passage_text)-1)
            self.N += 1

            passage_terms = set()
            for i in range(passage_len):
                term = self.get_term(passage_text, i)
                if term in passage_terms:
                    continue
                passage_terms.add(term)

                if term in term_docs_count:
                    term_docs_count[term] += 1
                else:
                    term_docs_count[term] = 1

        self.idfs = {}
        for term in term_docs_count:
            n = term_docs_count[term]
            self.idfs[term] = math.log((self.N-n+0.5)/(n+0.5) + 1)

        self.avgdl = sum(doc_lens)/len(doc_lens)


    def save_statistics(self, path_out):
        f_out = open(path_out, 'w')
        f_out.write(json.dumps({"docs_number": self.N})+'\n')
        f_out.write(json.dumps({"average_length": self.avgdl})+'\n')
        for term in self.idfs:
            f_out.write(json.dumps({
                "term": term,
                "idf": self.idfs[term]
                }, ensure_ascii=False)+'\n')


    def get_score(self, question, passage):
        question = self.tokenize(question)
        passage = self.tokenize(passage)
        if len(passage) == 0:
                return 0

        question_len = self.get_tokens_len(question)
        score = 0
        for i in range(question_len):
            term = self.get_term(question, i)
            tf = self.get_all_terms(passage).count(term)
            if tf > 0:
                score += self.idfs[term]*tf*(self.k1+1)/(tf+self.k1*(1-self.b+self.b*len(passage)/self.avgdl))
        return score