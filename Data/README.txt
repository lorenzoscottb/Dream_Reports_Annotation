+-----------------------------------------+
+Column-Names and content of the csv file +
+-----------------------------------------+

'gender': F/M 

'age': {'A', 'T', 'Y', 'YA'} (guess they stand for) {adult, teen, yung, yung-adult}

'type': seires/set, again not sure

'collection': source of the reposts (set of 6) 

'id': 'b-baseline', 'bea1', 'ed', 'emma', 'norms-f', 'norms-m'

'time': time window 

'date': date of the report (?) sometimes accopmpanied with age at time of rcord (?) 

'number': seems like id + year...

'report': dream report, raw string of text

'# words': ammount of tokens (words) in each report 

'Emotions': string (or '_' separated list) of Emotion-labels attributed to the dreamer

'# Emotions': number of emotions (lables) associate with each report 

'TSNE_x', 'TSNE_y': coordinates from the t-SNE reduction of the embeddings (PCA is preferable)

'PCA_x', 'PCA_y': coordinates from the PCA reduction of the embeddings

'Kmean_Cluster_6': clusters obtained with Sklearn Kmean algoritm, searching for 6 (Collection)

'Kmean_Cluster_2':  clusters obtained with Sklearn Kmean algoritm, searching for 2 (Gender) clusters

'2W_SA_label': POSITIVE/NEAGTIVE Sentiment Analysis labels**

'2W_SA_score': score (i.e. probability) assigned by the model to the selected label

'6W_SA_dict': lits of dictionaries*** with emotion and scores****



-------------------------------------------------------------------------------------------------
* To read the numpy file with the embeddings with python, and store it in the pandas 
datatframe (you can) use the following code

import pandas as pd
import numpy as np

dream_records = pd.read_csv("Dreams_with_embeddings.csv")
with open('BERT-Large-Cased_dream_records.npy', 'rb') as f:
    T_encoding = np.load(f)

dream_records["BERT-large-case-embeddings"] = list(T_encoding)

---------------------
** Model: distilbert-base-uncased-finetuned-sst-2-english

---------------------
*** Example of emotion dictionary from 6-way Sentiment analysis model
"""
Emotion Output:
[[
{'label': 'sadness', 'score': 0.0006792712374590337}, 
{'label': 'joy', 'score': 0.9959300756454468}, 
{'label': 'love', 'score': 0.0009452480007894337}, 
{'label': 'anger', 'score': 0.0018055217806249857}, 
{'label': 'fear', 'score': 0.00041110432357527316}, 
{'label': 'surprise', 'score': 0.0002288572577526793}
]]
"""

---------------------
**** Model: bhadresh-savani/distilbert-base-uncased-emotion
