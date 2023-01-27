_author_ = "lb540"

import random
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm 

def set_seed(seed: int):
    """Helper function for reproducible behavior to set the seed in ``random``, 
        ``numpy``, ``torch`` and/or ``tf`` (if installed).

    Args:
        seed (:obj:`int`): The seed to set
    """
    from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
    
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # ^^ safe to call this function even if cuda is not available
    if is_tf_available():
        import tensorflow as tf

        tf.random.set_seed(seed)
  

def get_PN_em(x, neg_em_list):
    if x in neg_em_list:
        return "Negative"
    elif x == "HA":
        return "Positive"
    else:
        return "Missing"


Coding_emotions = {
    "AN": "Anger",
    "AP": "Apprehension",
    "SD": "Sadness",
    "CO": "Confusion",
    "HA": "Happiness",
    
    "Missing": "Missing",
}


def Coding_em_binary_P(emotion_str):
    if emotion_str == "AP": 
        return "Positive"
    elif emotion_str == "Missing":
        return "Missing"
    else:
        return "Negative"

    
def get_SA_scores(dream_records, emotions="Dreamer", N_emotions="one", reference_Sentiment="POSITIVE"):
    
    """Function to collect the probability of observing the reference sentiment associated 
       with specific emotions, devided by Dream-Bank collection. In other words, extract the
       probability that a report is judged by the SA model as positive/negative, devided 
       by DreamBank emotions and classes

    Args:
        dream_records: dataframe containing reports and model predictions 
        N_emotions: string, number of emotions in each report [one, >1, all]
        reference_Sentiment: string, the referece sentiment fro the SA model
    """
    
    DRlbl_SAlbl = []
    for collection in tqdm(set(dream_records["collection"])):

        if N_emotions == "one":
            lcl_df          = dream_records[
                                    dream_records["collection"].isin([collection]) & 
                                    dream_records["# {} Emotions".format(emotions)].isin([1])       
                            ]

        elif N_emotions == "all":
            lcl_df          = dream_records[
                                    dream_records["collection"].isin([collection]) & 
                                    ~dream_records["# {} Emotions".format(emotions)].isin([0])       
                            ]
            
        elif N_emotions == ">1":
            lcl_df          = dream_records[
                                    dream_records["collection"].isin([collection]) & 
                                    ~dream_records["# {} Emotions".format(emotions)].isin([0,1])        
                            ]
        else:
            print("No such setting {}".formt(N_emotions))
            break

        lcl_DRlbl_SAlbl = {} # Dream-Report label vs Sent. Analysis label
        for emotions_seq, SA_lbl in lcl_df[["{} Emotions".format(emotions),"2W_SA_label"]].values:

            SA_lbl_to_int = 1 if SA_lbl == reference_Sentiment else 0

            for emt in emotions_seq.split("_"):

                local_lst = lcl_DRlbl_SAlbl.get(Coding_emotions[emt], [])
                local_lst.append(SA_lbl_to_int)
                lcl_DRlbl_SAlbl[Coding_emotions[emt]] = local_lst

        lcl_DRlbl_SAlbl = [
            [k,100*(sum(v)/len(v)), collection] for k,v in lcl_DRlbl_SAlbl.items()
        ]
        
        for trpl in lcl_DRlbl_SAlbl:
            DRlbl_SAlbl.append(trpl)
       
    return DRlbl_SAlbl


def get_general_sentiment(dream_records, Emotion_to_Score, emotions="General", method="diff"):
    
    DRlbl_SAlbl = [] # Dream-Report label vs Sent. Analysis label
    for collection in set(dream_records["collection"]):
        lcl_df = dream_records[
                        dream_records["collection"].isin([collection]) & 
                        ~dream_records["# {} Emotions".format(emotions)].isin([0])       
        ]

        for emotions_seq, SA_lbl, SA_scr in lcl_df[["{} Emotions".format(emotions), "2W_SA_label", "2W_SA_score"]].values:

            general_sentiment    = sum(list(map(lambda e: Emotion_to_Score[e], emotions_seq.split("_"))))
            general_predicted_SA = get_model_sentiment(SA_scr, SA_lbl, method=method)
            DRlbl_SAlbl.append([emotions_seq, general_sentiment, general_predicted_SA, collection])

    DRlbl_SAlbl = pd.DataFrame(DRlbl_SAlbl, columns=["Emotions", "General Sentiment", "Predicted Sentiment", "collection"])
    
    return DRlbl_SAlbl


def get_model_sentiment(SA_scr, SA_lbl, method="sign"):
    
    if method=="sign":
        return SA_scr if SA_lbl == "POSITIVE" else -SA_scr
    
    elif method=="diff":
        if SA_lbl == "POSITIVE":
            return SA_scr - (1- SA_scr)
        else:
            return -SA_scr + (1 - SA_scr) 
    else:
        print("No such method {}".format(method))
