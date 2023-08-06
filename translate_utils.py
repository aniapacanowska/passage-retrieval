from transformers import AutoModel, AutoTokenizer
from transformers import pipeline
from datasets import Dataset
from tqdm.auto import tqdm
from transformers.pipelines.pt_utils import KeyDataset
import json

tokenizer = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-pl-en')

def split_text(text):
    '''
    Split passages into pieces that fit in the model.
    '''
    MAX_LENGTH = 0.3*512 # Helsinki model does not work well with long passages
    sentences = [sentence+'.' for sentence in text.split('.') if sentence]
    text_pieces = []
    start = 0
    end = 1
    piece = ''
    while end <= len(sentences):
#         print(start, end)
        new_piece = ''.join(sentences[start:end])
        if len(tokenizer(new_piece)['input_ids']) > MAX_LENGTH:
            if end-start == 1:
                # There is a sentence that is longer than MAX_LENGTH - split it
                sentence_words = sentences[start].split()
                sentences = sentences[:start] + [' '.join(sentence_words[:len(sentence_words)//2]), ' '.join(sentence_words[len(sentence_words)//2:])] + sentences[start+1:]
                continue

            text_pieces.append(piece)
            # slices should not be disjoint - otherwise part of the answer might fall in one, and the rest in the other
            start = end-1
            piece = ''.join(sentences[start:end])
        else:
            piece = new_piece
            end += 1
    if start < end:
        text_pieces.append(piece)
    return text_pieces


def get_old_translations(path_translations):
    old_translations = {}
    for line in open(path_translations):
        translation = json.loads(line)
        old_translations[translation['passage_id']] = translation['passage_text']
    return old_translations


def get_passages_to_translate(old_translations, passages):
    passage_ids = {passage_id for passage_id in passages}
    old_ids = {old_id for old_id in old_translations}
    ids_to_translate = list(passage_ids - old_ids)
    texts_to_translate = [passages[passage_id] for passage_id in ids_to_translate]
    return (ids_to_translate, texts_to_translate)


def translate(ids, texts):
    dataset = Dataset.from_dict({'id':ids, 'text':texts})

    results = []
    pipe = pipeline('translation','Helsinki-NLP/opus-mt-pl-en', device=0)
    for result in tqdm(pipe(KeyDataset(dataset, 'text'), batch_size=16)):
        results.append(result)

    new_translations = {passage_id:[] for passage_id in ids}
    for passage_id, result in zip(ids, results):
        new_translations[passage_id].append(result[0]['translation_text'])

    return new_translations

def update_translations_file(path, new_translations):
    f_out = open(path, 'a')
    for passage_id in new_translations:
        f_out.write(json.dumps({
            'passage_id': passage_id,
            'passage_text': new_translations[passage_id]
        })+'\n')
    f_out.close()

# don't confuse allegro passage ids and question ids
def num_to_question_id(n, source='train'):
    if source == 'train':
        return 'q'+str(n)+'tr'
    if source == 'dev':
        return 'q'+str(n)+'d'
    if source == 'test-wiki':
        return 'q'+str(n)+'tw'
    if source == 'test-allegro':
        return 'q'+str(n)+'ta'
    if source == 'test-legal':
        return 'q'+str(n)+'tl'
    if source == 'mini':
        return 'q'+str(n)+'m'
    if source == 'testB-wiki':
        return 'q'+str(n)+'Bw'
    if source == 'testB-legal':
        return 'q'+str(n)+'Bl'
    if source == 'testB-allegro':
        return 'q'+str(n)+'Ba'

def question_id_to_num(id):
    return int(id[1:])