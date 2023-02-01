# Tuned model usage

First, make sure your envirment follws the given library requrments, found in the `SL_requrments.txt`. If your are using (as suggested) a conda env, you can do so by running. The, make sure to download the wights of the trained model [here](https://drive.google.com/file/d/16qROgqgQoOyImn4TUtm43zMflbJX89LO/view?usp=sharing). Note that all the experiments were run with python `3.9.12`.

`
conda install --file requirements.txt
`

Then, first, import the necessary dependecies 
```py
import torch, os
import pandas as pd
from tqdm import tqdm
import transformers
from transformers import AutoModel
from transformers import AutoConfig
from transformers import BertTokenizerFast
from SL_utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = str(5)
```

Then, set up the necessary data-related viariables
```py
Coding_emotions = {
    "AN": "Anger",
    "AP": "Apprehension",
    "SD": "Sadness",
    "CO": "Confusion",
    "HA": "Happiness",
}

emotions_list = list(Coding_emotions.keys())

test_sentences = [
    "In my dream I was follwed by the scary monster.",
    "I was walking in a forest, sorrounded by singing birds. I was in calm and peace."
]

test_sentences_target = len(test_sentences)*[[0, 0, 0, 0, 0]]
test_sentences_df     = pd.DataFrame.from_dict(
                {
                "report":test_sentences,
                "Report_as_Multilabel":test_sentences_target
                }
)
```

You can now set up the model and the necessary data-set format
```py
model_name   = "bert-large-cased"
model_config = AutoConfig.from_pretrained(model_name)
tokenizer    = BertTokenizerFast.from_pretrained(model_name, do_lower_case=False)
testing_set  = CustomDataset(test_sentences_df, tokenizer, max_length=512)

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

# Load the models' weights from the pre-treined model
model.load_state_dict(torch.load("pytorch_model.bin"))
model.to("cuda")
```

Collect the model's prediction/classification
```py
outputs, targets, ids = validation(model, testing_loader, device="cuda", return_inputs=True)

corr_outputs    = np.array(outputs) >= 0.5 
corr_outputs_df = pd.DataFrame(corr_outputs, columns=emotions_list)
corr_outputs_df = corr_outputs_df.astype(int)

corr_outputs_df["report"] = decoded_ids = [decode_clean(x, tokenizer) for x in tqdm(ids)]
```

Lastly, inspect the predictions
```py 
corr_outputs_df
```
```
AN	AP	SD	CO	HA	report
0	0	0	0	1	I was walking in a forest, sorrounded by sing...
0	1	0	0	0	In my dream I was follwed by the scary monste..
```
