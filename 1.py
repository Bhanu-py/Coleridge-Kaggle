import pandas as pd
import os
import json
from tqdm.auto import tqdm
import re
import spacy

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# x = os.listdir("E:/kaggle/Coleridge/coleridgeinitiative-show-us-the-data/train/")
# with open(f'coleridgeinitiative-show-us-the-data/train/{x[0]}') as f:
#     data = json.load(f)
#     print(json.dumps(data, indent=4, sort_keys=True))
# filename = "000e04d6-d6ef-442f-b070-4309493221ba"
# df_json = pd.read_json("/coleridgeinitiative-show-us-the-data/train/" + str(filename) + ".json")

# df_train = pd.read_csv("E:\kaggle\Coleridge\coleridgeinitiative-show-us-the-data/train.csv")
#
#
# def data(filename):
#     df_json = pd.read_json("E:/kaggle/Coleridge/coleridgeinitiative-show-us-the-data/train/"+str(filename)+".json")
#     text = "".join(row['text'] for _, row in df_json.iterrows())
#     return text
#
#
# def clean_text(txt):
#     return re.sub('[^A-Za-z0-9]+', ' ', str(txt).lower())
#
#
# tqdm.pandas()
# df_train['Text'] = df_train['Id'].progress_apply(data)
#
# tqdm.pandas()
# df_train['Text'] = df_train['Text'].progress_apply(clean_text)
# df_train.to_csv("E:\kaggle\Coleridge\coleridgeinitiative-show-us-the-data/df_train.csv", index=False, header=True)

df_train = pd.read_csv("E:/kaggle/Coleridge/coleridgeinitiative-show-us-the-data/df_train.csv")
# print(df_train.head(5))

nlp = spacy.load('en')
tokens = []
for doc in nlp.pipe(df_train['Text'], batch_size=50):
    if doc.is_parsed:
        tokens.append([n.text for n in doc])
    else:
        tokens.append(None)
print(tokens)



