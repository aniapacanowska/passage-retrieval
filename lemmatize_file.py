'''
Lemmatizers: Morfeusz2, spaCy, mixed, freq.
'''

import json
import sys
import os
from multiprocessing import Pool
import morfeusz2
import spacy

class SpaCyLemmatizer:
    def __init__(self):
        self.lemmatizer = spacy.load('pl_core_news_sm')

    def lemmatize(self, text):
        return ' '.join([token.lemma_ for token in self.lemmatizer(text)])


class MorfeuszLemmatizer:
    def __init__(self):
        self.morf = morfeusz2.Morfeusz(generate=False)

    def get_lemma_list(self, text):
        analysis = self.morf.analyse(text)
        lemmas = []
        for node in analysis:
            word_id = node[0]
            lemma = node[2][1].split(':')[0] # Add only the lemma without the identifier after ':'.
            if len(lemmas) == word_id:
                lemmas.append([lemma])
            else:
                # One word can have multiple lemmas with the same spelling, but meaning different things (homonyms)
                # For example 'nie:C', 'nie:I' 'nie:T'. Add only one of such lemmas.
                if lemma not in lemmas[word_id]:
                    lemmas[word_id].append(lemma)
        return lemmas

    def lemmatize(self, text):
        lemmas = self.get_lemma_list(text)
        return ' '.join([' '.join(word_lemmas) for word_lemmas in lemmas])

class MixedLemmatizer:
    '''
    Morfeusz lemmatizer does not choose the correct lemma for the context.
    SpaCy lemmatizer sometimes creates words that do not exist.
    Try to use both:
    If the spaCy lemma is one of the Morfeusz lemmas, take it.
    Otherwise use the most popular Morfeusz lemma.
    '''
    def __init__(self, morfeusz_passages='data/wiki-trivia/passages-morfeusz.jl', load_spacy=True):
        self.morfeusz = MorfeuszLemmatizer()
        self.freq = self.get_lemma_frequency(morfeusz_passages)
        if load_spacy:
            self.spacy = spacy.load('pl_core_news_sm')

    def get_lemma_frequency(self, morfeusz_passages):
        # get the frequency of each lemma in the corpus lemmatized with Morfeusz
        freq = {}
        for line in open(morfeusz_passages):
            text = json.loads(line)['text']
            for word in text.split():
                freq[word] = freq.get(word, 0) + 1
        return freq

    def get_most_popular_lemma(self, lemmas):
        best_lemma = ''
        best_count = -1
        for lemma in lemmas:
            count = self.freq.get(lemma, 0)
            if count > best_count:
                best_lemma = lemma
                best_count = count
        return best_lemma

    def lemmatize(self, text):
        lemmas = []
        for token in self.spacy(text):
            lemma = token.lemma_
            morfeusz_lemmas = self.morfeusz.get_lemma_list(token.text)
            if len(morfeusz_lemmas) == 0: # invalid token, such as ' '
                continue

            # take spacy lemma for numbers, interpunction and abbrevietions
            # words under 3 letters are likely abbrevietions
            if lemma.isalpha() and len(lemma) > 3:  
                if lemma.lower() not in morfeusz_lemmas[0]:
                    lemma = self.get_most_popular_lemma(morfeusz_lemmas[0])
            lemmas.append(lemma)
        return ' '.join(lemmas)
    

class FrequencyLemmatizer:
    '''
    Use the most popular Morfeusz lemma.
    '''
    def __init__(self):
        self.mixed = MixedLemmatizer(load_spacy=False)
        

    def lemmatize(self, text):
        lemmas = []
        for word_lemmas in self.mixed.morfeusz.get_lemma_list(text):
            lemmas.append(self.mixed.get_most_popular_lemma(word_lemmas))
        return ' '.join(lemmas)


class JSONFile:
    def __init__(self, out_path):
        self.f_out = open(out_path, 'w')

    def read_lines(self, in_path):
        return [json.loads(line) for line in open(in_path)]
    
    def write_lines(self, lines):
        for line in lines:
            self.f_out.write(json.dumps(line, ensure_ascii=False)+'\n')
    
    def close(self):
        self.f_out.close()

class CSVFile:
    def __init__(self, out_path):
        self.f_out = open(out_path, 'w')
        self.first = True

    def read_lines(self, in_path):
        return open(in_path).read().split('\n')
    
    def write_lines(self, lines):
        for line in lines:
            self.f_out.write(line+'\n')
    
    def close(self):
        # Remove trailing newline
        self.f_out.seek(self.f_out.tell()-1, os.SEEK_SET)
        self.f_out.truncate()
        self.f_out.close()
        

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('usage: python lemmatize_file.py in_path out_path lemmatizer')
        exit()

    path = sys.argv[1]
    out_path = sys.argv[2]

    if sys.argv[3] == 'spacy':
        lemmatizer = SpaCyLemmatizer()
    elif sys.argv[3] == 'morfeusz':
        lemmatizer = MorfeuszLemmatizer()
    elif sys.argv[3] == 'mixed':
        lemmatizer = MixedLemmatizer()
    elif sys.argv[3] == 'frequency':
        lemmatizer = FrequencyLemmatizer()
    else:
        print('Unknown lemmatizer. Can be one of: spacy, morfeusz, mixed, frequency')
        exit()


    # This function is used in Pool.map(), has to be picklable
    def lemmatize_line_json(line):
        line['text'] = lemmatizer.lemmatize(line['text'])
        return line

    # This function is used in Pool.map(), has to be picklable
    def lemmatize_line_csv(line):
        tokens = line.split()
        if len(tokens) < 2:
            return line
        domain = tokens[0]
        text = ' '.join(tokens[1:])
        return domain+'\t'+lemmatizer.lemmatize(text)

    if path[-3:] == '.jl':
        file = JSONFile(out_path)
        lemmatize_func = lemmatize_line_json
    else:
        file = CSVFile(out_path)
        lemmatize_func = lemmatize_line_csv


    lines = file.read_lines(path)
    N = 50000
    for i in range(0, len(lines), N): # process the dataset in smaller batches not to lose everything in case of an error
        with Pool() as p:
            lemmatized = p.map(lemmatize_func, lines[i: min(i+N, len(lines))])
        file.write_lines(lemmatized)
    file.close()