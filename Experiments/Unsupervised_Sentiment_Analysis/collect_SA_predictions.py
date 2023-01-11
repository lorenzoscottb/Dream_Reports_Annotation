_author_ = "lb540"

import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from SA_utils import set_seed
from transformers import pipeline


parser = argparse.ArgumentParser()
parser.add_argument(
    "--file", 
    type=str, 
    default="path_to_file.csv",
    help="path to the .csv file, must include a report and # words column"
)
parser.add_argument(
    "--report_column", 
    type=str, 
    default="report",
    help="name of the column in the DF/cvs that contains the dream reports"
)
parser.add_argument(
    "--words_column", 
    type=str, 
    default="# words",
    help="name of the column in the DF/cvs that contains the no. words per report"
)
parser.add_argument(
    "--model", 
    type=str, 
    default="distilbert-base-uncased-finetuned-sst-2-english",
    help="name of the model to (download and) use"
)
parser.add_argument(
    "--task", 
    type=str, 
    default="text-classification",
    help="name of the task for the Hugging-Face pipeline"
)
parser.add_argument("--max_len",  type=int, default=512, help="max len of the toneizer")
parser.add_argument("--GPU", type=int, default=0, help="which GPU to use (if any)")
parser.add_argument("--truncate", type=bool, default=True, help="truncate input")
parser.add_argument("--out_save", type=bool, default=False, help="model output format")
parser.add_argument("--seed", type=int, default=31, help="random seed")

args = parser.parse_args()

# Set up main variables
FILE_NAME  = args.file
REPORT_COL = args.report_column
WORDS_COL  = args.words_column

MODEL_NAME = args.model
TASK       = args.task
MAX_LEN    = args.max_len
DEVICE     = args.GPU
TRUNCATION = args.truncate
OUT_file   = args.out_save # Note: this sets the model's output to majority class. Set to True if  
                           # using non-binary classification.

#  set the random seed
seed = args.seed
set_seed(seed)


# Load the dataset in CSV
dream_records = pd.read_csv(
            FILE_NAME
)


# Set up the pipeline to obtain model's encodings
sent_pipeline = pipeline(
    TASK,
    model=MODEL_NAME, 
    return_all_scores=OUT_file, 
    truncation=TRUNCATION, 
    max_length=MAX_LEN, 
    device=DEVICE,
)


# Get the reports as list
dream_records = dream_records[~dream_records[WORDS_COL].isin([0])] # removes empty items
data          = dream_records[REPORT_COL].tolist()


# Get Model predictions from pipeline
predictions = [
    list(sent_pipeline(report)[0].values())
    for report in tqdm(data)
]


# Update the DF with model's predictions
dream_records["2W_SA_label"], dream_records["2W_SA_score"] = zip(*predictions)


# Save new df as .csv
dream_records.to_csv(
    "Dream_Reports_with-{}_predictions.csv".format(MODEL_NAME), 
    index=False,
)
