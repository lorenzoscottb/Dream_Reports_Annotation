_author_ = "lb540"

import os, argparse, torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn import metrics

from SL_utils import *
from transformers import BertTokenizerFast, BertForSequenceClassification

# argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "--file", 
    type=str, 
    default="../Data/Reports_with_Dreamer_and_General_Emotions.csv",
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
    default="bert-large-cased",
    help="name of the model to (download and) use"
)
parser.add_argument("--Emotion_Set", type=str, default="General", help="General / Dreamer")
parser.add_argument("--max_len",  type=int, default=512, help="max len of the toneizer")
parser.add_argument("--epochs", type=int, default=10, help="training epochs")
parser.add_argument("--fold", type=int, default=5, help="n. of fold for the experiment")
parser.add_argument("--lr", type=float, default=1e-05, help="learning rate")
parser.add_argument("--batch_train", type=int, default=8, help="batch size for training")
parser.add_argument("--batch_eval", type=int, default=2, help="batch size for evaluation")
parser.add_argument("--froze_LLM", type=bool, default=False, help="don't train the LLM")
parser.add_argument("--GPU", type=int, default=0, help="which GPU to use (if any)")
parser.add_argument("--train_device", type=str, default="cuda", help="cpu / cuda")
parser.add_argument("--truncate", type=bool, default=True, help="truncate input")
parser.add_argument("--seed", type=int, default=31, help="random seed")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)

# Set-up main variables
FILE_NAME  = args.file
REPORT_COL = args.report_column
WORDS_COL  = args.words_column
EMOTIONS   = args.Emotion_Set

seed = args.seed
set_seed(seed)



# Data set-up and extraction
dream_records = pd.read_csv(FILE_NAME)

Coding_emotions = {
    "AN": "Anger",
    "AP": "Apprehension",
    "SD": "Sadness",
    "CO": "Confusion",
    "HA": "Happiness",
    "Missing": "Missing",
}

emotions_list = list(Coding_emotions.keys())
emotions_list.remove("Missing")

EMOTION_SET = "{} Emotions".format(EMOTIONS)
print("Collect data. Emotion Set : {}".format(EMOTION_SET))
report_as_multi_label = []
for rprt_emtn_lst in tqdm(dream_records[EMOTION_SET]):
    lcl_report_as_multi_label = []
    for emotion_acronim in emotions_list:
        if emotion_acronim in rprt_emtn_lst:
            lcl_report_as_multi_label.append(1)
        else:
            lcl_report_as_multi_label.append(0)
    report_as_multi_label.append(lcl_report_as_multi_label)

dream_records["Report_as_Multilabel"] = report_as_multi_label

final_df_dataset = dream_records[~dream_records["# {}".format(EMOTION_SET)].isin([0])].reset_index(drop=True)

final_df_dataset = final_df_dataset[["report", "Report_as_Multilabel", "collection"]]



# Model's set-up and training
model_name        = args.model
max_length        = args.max_len
device            = args.train_device
epochs            = args.epochs
train_batch_size  = args.batch_train
valid_batch_size  = args.batch_eval
learning_rate     = args.lr
froze_model_layer = args.froze_LLM 
K_fold            = args.fold

tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=False)

SCORES, LOSSES, DATAFRAMES = [], [], []
for collection in set(final_df_dataset["collection"]):

    print('Held-out collection {}'.format(collection))
    train_loader, testing_loader = get_collection_Fold(
        final_df_dataset, 
        collection, 
        tokenizer, 
        train_batch_size, 
        valid_batch_size, 
        max_length=max_length,
    )

    # makes sure the seed for model init is the same
    set_seed(seed, set_random=False) 
    
    # Set Model and Optmizer (for each fold)
    model_tuned = BERTClass(
        model_name=model_name, 
        n_classes=len(emotions_list), 
        freeze_BERT=froze_model_layer
    )

    optimizer_tuned = torch.optim.Adam(
    params=model_tuned.parameters(), 
    lr=learning_rate
    ) 
    
    model_tuned.to(device)
    
    #Train and test the model
    for ep in range(epochs):
        train_losses = train(
            ep, 
            model_tuned, 
            train_loader, 
            optimizer_tuned, 
            return_losses=True,
            device=device,
        )
        LOSSES.append(train_losses)
    
    print("Testing the Model")
    outputs, targets = validation(model_tuned, testing_loader, device=device)
    outputs = np.array(outputs) >= 0.5    
    
    # send the model back to cpu and delete
    model_tuned.to("cpu")
    del model_tuned 
    
    # Compute and print results
    results_df = pd.DataFrame.from_dict(
        metrics.classification_report(
            targets,
            outputs,
            target_names=emotions_list,
            zero_division=0,
            output_dict=True,
        ), 
        orient='index',
    ).round(2)
    
    results_df["collection"] = [collection]*len(results_df)
    DATAFRAMES.append(results_df)
    
    f1_samp_av = results_df.at["samples avg", "f1-score"]
    print("F1 samples avg.: {}".format(f1_samp_av), end = "\n")
    SCORES.append(f1_samp_av)
    
# Save results as csv
DF_all = pd.concat(DATAFRAMES)
for clmn in ["precision","recall","f1-score"]:
  DF_all[clmn] = DF_all[clmn].apply(lambda x: 100*x)
DF_all.to_csv("Ablation_results_{}_{}.csv".format(EMOTIONS, model_name))
