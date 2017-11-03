import requests
import pandas as pd
import numpy as np
import time

dataset = pd.read_csv('MovieGenre.csv')
poster_names = dataset.iloc[:, 2].values
poster_genres = dataset.iloc[:, 4].values
poster_links = dataset.iloc[:, 5].values

index_flag = []


for x in range(0, 10000):
    print(x)
    if x <= 8000:
        try:
            response = requests.get(poster_links[x])
            if response.status_code == 200:
                file_name = "Dataset\\training_set\\" + poster_names[x] + ".jpg"
                with open(file_name, "wb") as f:
                    f.write(response.content)
        except:
            index_flag.append(x)
    elif x > 8000:
        try:
            response = requests.get(poster_links[x])
            if response.status_code == 200:
                file_name = "Dataset\\test_set\\" + poster_names[x] + ".jpg"
                with open(file_name, "wb") as f:
                    f.write(response.content)
        except:
            index_flag.append(x)
    time.sleep(1)        