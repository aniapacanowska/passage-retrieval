import json

def read_rows(path):
    rows = []
    for line in open(path):
        rows.append(json.loads(line))
    return rows

def write_json_format(path_out, rows):
    f_out = open(path_out, 'w')
    for row in rows:
        f_out.write(json.dumps(row, ensure_ascii=False)+'\n')