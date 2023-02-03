_author_ = "lb540"

import random
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm 
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import transformers 

#################################################

# General Utils

#################################################

def set_seed(seed: int, set_random=True):
    """Helper function for reproducible behavior to set the seed in ``random``, 
        ``numpy``, ``torch`` and/or ``tf`` (if installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
    
    if set_random:
        random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # ^^ safe to call this function even if cuda is not available
    if is_tf_available():
        import tensorflow as tf

        tf.random.set_seed(seed)
        
def decode_clean(x, tokenizer):
    s = tokenizer.decode(x).replace("[PAD]", "").replace("[CLS]", "").replace("[SEP]", "")
    return s

#################################################

# Data manipulation & set-up

#################################################

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_length=512):
        self.tokenizer = tokenizer                      # the Tokenizer model
        self.data      = dataframe                      # the full dataset
        self.report    = dataframe.report               # the text data (i.e., the reports)
        self.targets   = self.data.Report_as_Multilabel # labels' list to classify
        self.max_len   = max_length                     # max length fro truncation

    def __len__(self):
        return len(self.report)

    def __getitem__(self, index):
        report = str(self.report[index])
        report = " ".join(report.split())

        inputs = self.tokenizer.encode_plus(
            report,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }
    
def get_Fold(final_df_dataset, tokenizer, k_seed, train_batch_size, valid_batch_size, max_length=512, train_size=0.8):
    
    train_dataset = final_df_dataset.sample(frac=train_size, random_state=k_seed)
    test_dataset  = final_df_dataset.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    training_set = CustomDataset(train_dataset, tokenizer, max_length=max_length)
    testing_set  = CustomDataset(test_dataset, tokenizer, max_length=max_length)

    train_params = {
        'batch_size': train_batch_size,
        'shuffle': True,
        'num_workers': 0
    }

    test_params = {
        'batch_size': valid_batch_size,
        'shuffle': True,
        'num_workers': 0
    }

    training_loader = DataLoader(training_set, **train_params)
    testing_loader  = DataLoader(testing_set, **test_params)
    
    return training_loader, testing_loader

def get_collection_Fold(final_df_dataset, collection, tokenizer, train_batch_size, valid_batch_size, max_length=512):
    
    train_dataset = final_df_dataset[~final_df_dataset["collection"].isin([collection])]
    test_dataset  = final_df_dataset[final_df_dataset["collection"].isin([collection])]

    train_dataset = train_dataset.reset_index(drop=True)
    test_dataset  = test_dataset.reset_index(drop=True)

    training_set = CustomDataset(train_dataset, tokenizer, max_length)
    testing_set  = CustomDataset(test_dataset, tokenizer, max_length)

    train_params = {
        'batch_size': train_batch_size,
        'shuffle': True,
        'num_workers': 0
    }

    test_params = {
        'batch_size': valid_batch_size,
        'shuffle': True,
        'num_workers': 0
    }

    training_loader = DataLoader(training_set, **train_params)
    testing_loader  = DataLoader(testing_set, **test_params)
    
    return training_loader, testing_loader

#################################################

# Architectures

#################################################

class BERTClass(torch.nn.Module):
    def __init__(self, model_name, n_classes, freeze_BERT=False, layer=-1, idx=0):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained(model_name)
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(1024, n_classes)
        self.layer = layer
        self.idx   = idx  
        # Froze the weight of model aside of the classifier
        if freeze_BERT:
            print("Freezing the layer of BERT model")
            for name, param in self.l1.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False
                    
    def forward(self, ids, mask, token_type_ids):
        output_1 = self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids)
        output_1 = output_1.last_hidden_state[:, -1, :]
        output_2 = self.l2(output_1)
        output   = self.l3(output_2)
        return output

class BERT_PTM(transformers.PreTrainedModel):
    def __init__(self, config, model_name, n_classes, freeze_BERT=False, layer=-1, idx=0):
        super(BERT_PTM, self).__init__(config)
        self.l1 = transformers.BertModel.from_pretrained(model_name)
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(1024, n_classes)
        self.layer = layer
        self.idx   = idx  
        # Froze the weight of model aside of the classifier
        if freeze_BERT:
            print("Freezing the layer of BERT model")
            for name, param in self.l1.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False
                    
    def forward(self, ids, mask, token_type_ids):
        output_1 = self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids)
        output_1 = output_1.last_hidden_state[:, -1, :]
        output_2 = self.l2(output_1)
        output   = self.l3(output_2)
        return output

#################################################

# Training & Validation

#################################################

def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

def train(epoch, model, training_loader, optimizer, return_losses=False, device="cuda"):
    Losses = []
    model.train()
    for _, data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.float)

        outputs = model(ids, mask, token_type_ids)

        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        if _%5000==0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')
            Losses.append(loss.item())
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if return_losses:
        return Losses
    
def validation(model, testing_loader, device="cuda"):
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model(ids, mask, token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets
