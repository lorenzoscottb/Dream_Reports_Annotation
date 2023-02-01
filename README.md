# Automatic Dream–Reports Annotation with Large Language Models (LLMs)

This repository contains the code, results, and analysis of the experiment on the automatic annotation of dream-reports. The annotation process is largely based on pre-trained largel language models (implemented maninly via Hugging Face 🤗).

# Content
### Experiments

The work revolvs around two main set of experiments:

- Unsupervised sentiment analysis
- Supervised text classification

Code for each set of experiments can be found in the dedicated folders inside the `Experiments` folders.

### Analysis

The analysis of the collected results can be found in the dedicated jupiter notebooks.

# Trained Models
## Main model
The main model can be downloaded and used via the `collect_predictions.py` script from the `Experiments/Supervised_Learning`.

## Secondary 🤗 models 
Together with the main deployed model, we trained and open-sourced two more models, which are two LLMs tuned as `multi-class` classifiers solely using the the standard 🤗 trainer pipeline. Despite Achiving a lower performance, these models posses other desireble features. First, they can be directly dowladed and used via 🤗 ```transformers``` library. Secondly, one of the released model can annotate dreams from 94 languages, whle the second is based on a (engluhs only) smaller encoder, hence rquireing less computational power. 

### Usage
[Large-Multilingual](https://huggingface.co/lorenzoscottb/xlm-roberta-large-DreamBank)
```py
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("lorenzoscottb/xlm-roberta-large-DreamBank")

model = AutoModelForSequenceClassification.from_pretrained("lorenzoscottb/xlm-roberta-large-DreamBank")
```

[Small–English only](https://huggingface.co/lorenzoscottb/bert-base-cased-DreamBank)
```py
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("lorenzoscottb/bert-base-cased-DreamBank")

model = AutoModelForSequenceClassification.from_pretrained("lorenzoscottb/bert-base-cased-DreamBank")
```
### Test via Spaces

You can also directly test our 🤗 models in an API fasion via the dedicated Hugging Face Space [here](https://huggingface.co/spaces/lorenzoscottb/DSA-II)

# Requirments

See the `*_requirements.txt` files for each set of experiment.

# Acknowledgements

## Data
### Labelled Dream Reports
As specified in the paper, the labelled data is frely availeb fro consulation via the Dream Bank website. The labelled data adopted in the supervised experiment consists of an xlm version of Dream Bank availabe upon request to the Dream Bank team.

### Full Dream Bank
The unlabelled-data analysis was possible thanks to the [Dream Bank (and pre-scraped) data scraper](https://github.com/mattbierner/DreamScrape) from Matt Bierner.
