'''
Get GPT3 answers using OpenAI API.
'''


import os
import json
import openai
from ratelimiter import RateLimiter


def read_rows(path):
    rows = []
    for line in open(path):
        rows.append(json.loads(line))
    return rows

def read_answers(path):
    answers = {}
    for line in open(path):
        row = json.loads(line)
        answers[row["question"]] = row["answer"]
    return answers


def write_json_format(path_out, rows):
    f_out = open(path_out, 'w')
    for row in rows:
        f_out.write(json.dumps(row, ensure_ascii=False)+'\n')


def get_ending(model, path_dataset):
    if model == "gpt-3.5-turbo":
        ending = " "
        if path_dataset.endswith("legal.jl"):
            ending += "W Polse. "
        if path_dataset.endswith("allegro.jl"):
            ending += "W allegro. "
        return ending+"Shortest answer in english."
    else:
        return " Shortest answer. NOT unknown."



model = "gpt-3.5-turbo" # text-davinci-003 / gpt-3.5-turbo
path_dataset = "test-B/legal.jl" 
path_answers = "answers.jl"
ending = get_ending(model, path_dataset)

rows = read_rows(path_dataset)
answers = read_answers(path_answers)
file_answers = open(path_answers, "a")

rate_limiter = RateLimiter(max_calls=1, period=4)
failed_retries = {}
for row in rows:
    if model == 'gpt-3.5-turbo':
        question = row["question_text"]
    else:
        question = row["question_translated"] # davinci needs questions in english
    if question not in answers:
        with rate_limiter:
            try:
                if model == "gpt-3.5-turbo":
                    response = openai.ChatCompletion.create(
                        model=model,
                        messages = [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role":"user", "content": question+ending}
                        ],
                        temperature=0.1
                        )
                    answers[question] = response['choices'][0]['message']['content']
                else:
                    response = openai.Completion.create(model=model, prompt=question+ending, temperature=0.1, max_tokens=30)
                    answers[question] = response['choices'][0]['text'][2:]
                file_answers.write(json.dumps({"question": question, "answer": answers[question]}, ensure_ascii=False)+'\n') # save answers to file in case something goes wrong during execution
                print(row['question_text'], question, ":", answers[question])
            except Exception as e:
                print("Error generating answer for question: " + question)
                print(e)
    if question in answers:
        if model == "gpt-3.5-turbo":
            row["chatgpt_answer"] = answers[question]
        else:
            row["gpt3_answer"] = answers[question]

write_json_format(path_dataset, rows)