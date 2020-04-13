import pandas as pd 
import numpy as np
import urllib, json
from urllib.request import urlopen

filename = 'ORES_test_data.csv'
df = pd.read_csv(filename)

url_1 = "https://ores.wikimedia.org/v3/scores/enwiki/"
url_2 = "/damaging?features"

# rev_id = "644933637"
# url = url_1 + rev_id + url_2
# response = urlopen(url)
# data = json.loads(response.read())
# feature = (data["enwiki"]["scores"][rev_id]["damaging"]["features"])

# df = pd.concat([df, pd.DataFrame(columns=list(feature.keys()))])
# df.to_csv(filename, index=False)

def update_row_with_dict(dictionary, dataframe, index):
    for key in dictionary.keys():
        dataframe.loc[index, key] = dictionary.get(key)

r, c = df.shape
print(r, c)
for i in range(r):

    try:
        print(str(i) + " / 5000")
        rev_id = str(int(df.rev_id[i]))
        url = url_1 + rev_id + url_2
        print(url)
        response = urlopen(url)
        data = json.loads(response.read())
        feature = (data["enwiki"]["scores"][rev_id]["damaging"]["features"])

        update_row_with_dict(feature, df, i)

    except:
        print("no feature for " + rev_id)

    df.to_csv(filename, index=False)

df.to_csv(filename, index=False)
