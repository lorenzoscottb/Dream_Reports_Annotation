_author_ = "lb540"


import torch, os
from tqdm import tqdm
import pandas as pd
import transformers
from transformers import AutoModel
from transformers import AutoConfig
from transformers import BertTokenizerFast
import seaborn as sns
from SL_utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = str(5)


Coding_emotions = {
    "AN": "Anger",
    "AP": "Apprehension",
    "SD": "Sadness",
    "CO": "Confusion",
    "HA": "Happiness",
}

emotions_list = list(Coding_emotions.keys())

# Load full Dream Bank (i.e., not just the labelled data)
DreamBank   = pd.read_csv("../Data/dreams_min.csv")

# Set-up data as for the training and testing
vet_dreams = DreamBank[DreamBank["dream_series_id"].isin(["vietnam_vet"])]["content"].tolist()
vet_target = len(vet_dreams)*[[0, 0, 0, 0, 0]]
vet_dreams_df = pd.DataFrame.from_dict(
                {
                "report":vet_dreams,
                "Report_as_Multilabel":vet_target
                }
)


model_name = "bert-large-cased"
model_config = AutoConfig.from_pretrained(model_name)

tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=False)

testing_set = CustomDataset(vet_dreams_df, tokenizer, max_length=512)

test_params = {
    'batch_size': 2,
    'shuffle': True,
    'num_workers': 0
}

testing_loader  = DataLoader(testing_set, **test_params)

model = BERT_PTM(
    model_config,
    model_name=model_name, 
    n_classes=len(emotions_list), 
    freeze_BERT=False,
)

model.load_state_dict(torch.load("model/pytorch_model.bin"))

print("Collecting Predictions")
model.to("cuda")
outputs, targets, ids = validation(model, testing_loader, device="cuda", return_inputs=True)
corr_outputs = np.array(outputs) >= 0.5 

model.to("cpu")

corr_outputs_df = pd.DataFrame(corr_outputs, columns=emotions_list)
corr_outputs_df = corr_outputs_df.astype(int)
corr_outputs_df["# Emotions"] = [sum(v) for v in corr_outputs_df.values]
corr_outputs_df["report"] = decoded_ids = [decode_clean(x, tokenizer) for x in tqdm(ids)]

corr_outputs_df.to_csv("Model_Predictions.csv", index=False)
