import os
import csv
import json

import pandas as pd

def load(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
    
        data = []
        for line in reader:
            d = {
                'context': line[1],
                'question': line[2],
                'answer': line[3],
            }
            data.append(d)
    return data

def test_load(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
    
        data = []
        for line in reader:
            d = {
                'id': line[0],
                'context': line[1],
                'question': line[2],
            }
            data.append(d)
    return data

def to_pandas(data_path, data_list):
    dt = pd.read_csv(data_path)
    dt['answer'] = data_list
    dt = dt[['id','answer']]
     
    return dt

def cumulative_pandas(file_name, data):
    df = pd.DataFrame(data)
    if not os.path.exists(file_name):
        df.to_csv(file_name, index=False, mode='w')
    else:
        df.to_csv(file_name, index=False, mode='a', header=False)