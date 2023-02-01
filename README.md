# Automatic Dream–Reports Annotation with Large Language Models

This repository contains the code, results, and analysis of the experiment on the automatic annotation of dream-reports. The annotation process is largely based on pre-trained largel language models (LLMs), implemented maninly via Hugging Face 🤗.

# Content
## Experiments

The work revolvs around two main set of experiments:

- Unsupervised sentiment analysis
- Supervised text classification

Code for each set of experiments can be found in the dedicated folders inside the `Experiments` folders.

## Analysis

The analysis of the collected results can be found in the dedicated jupiter notebooks.

## Results 

The results and predictions collected in all the experiments. 

# Trained Models
## Main model
A link to downloaded the weights of the main model can be found in the `Experiments/Supervised_Learning` section, together with the code used to tune the the model, collect the predictions preseted in the paper, as well as a basic usage example.

### Download and usage 
You can find a use-case example of the main model, together with the link to download the weights [here](https://github.com/lorenzoscottb/Dream_Reports_Annotation/tree/main/Experiments/Supervised_Learning)

## Secondary 🤗 models 
While the main model achieves the best and most stable results, it is based on a custom architecture. Hence, setting up a classification pipeline requires more coding dependecies (see link above). For this reason, together with the main deployed model, we trained and open-sourced two more models, which are two LLMs "simply" tuned as `multi-class` classifiers, and solely using the the standard 🤗 trainer pipeline. Despite Achiving a lower performance, these models posses other desireble features. To start, they can be directly dowladed and used via the 🤗 ```transformers``` library. Moreover, one can annotate dreams in 94 languages, while the other is based on a (English-only) "small" LLM encoder, hence rquireing signficantly less computational power. 

### Usage
Select a tokenizer and a model between 

[Large-Multilingual](https://huggingface.co/lorenzoscottb/xlm-roberta-large-DreamBank)
```py
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("lorenzoscottb/xlm-roberta-large-DreamBank")
model     = AutoModelForSequenceClassification.from_pretrained("lorenzoscottb/xlm-roberta-large-DreamBank")
```

[Small–English only](https://huggingface.co/lorenzoscottb/bert-base-cased-DreamBank)
```py
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("lorenzoscottb/bert-base-cased-DreamBank")
model     = AutoModelForSequenceClassification.from_pretrained("lorenzoscottb/bert-base-cased-DreamBank")
```

Setup the classification pipleine and the dreams to classify
```py
from transformers import pipeline

# get some dream to classify
test_sentences = [
    "In my dream I was follwed by the scary monster.",
    "I was walking in a forest, sorrounded by singing birds. I was in calm and peace."
]

# set up the pipeline
classifier = pipeline(
    task="text-classification", 
    model=model, 
    tokenizer=tokenizer,
    return_all_scores=True, # Fasle to get above-threshold classes only
)

# get the model's classification
predictions = classifier(test_sentences)

# print the predictions' dictionaries (i.e., the probability associated with each Hall & Van de Castle emotion:
# anger (AN) apprehension (AP), sadness (SD), confusion (CO), happiness (HA)
predictions
>>> [[{'label': 'AN', 'score': 0.021188955754041672},
>>> {'label': 'AP', 'score': 0.8773345351219177},
>>> {'label': 'SD', 'score': 0.010038740932941437},
>>> {'label': 'CO', 'score': 0.0854405090212822},
>>> {'label': 'HA', 'score': 0.03229339420795441}],
>>> [{'label': 'AN', 'score': 0.007893212139606476},
>>> {'label': 'AP', 'score': 0.08208194375038147},
>>> {'label': 'SD', 'score': 0.03895331546664238},
>>> {'label': 'CO', 'score': 0.032238591462373734},
>>> {'label': 'HA', 'score': 0.9570998549461365}]]
````
### Query via Spaces

To get an ide of the classification abilitied of the two 🤗 models you can also directly query them via the [Hugging Face Space](https://huggingface.co/spaces/lorenzoscottb/DSA-II) built on top of them. You can also you the space to check if the language your reports are in is included in the multi-lingual model.

# Requirments

If you want to re-run the code of any of our experiment, please make sure to chek the `*_requirements.txt` files for each set of experiment.

# Acknowledgements

## Data
### Labelled Dream Reports
As specified in the paper, the labelled data is frely availeb fro consulation via the Dream Bank website. The labelled data adopted in the supervised experiment consists of an xlm version of Dream Bank availabe upon request to the Dream Bank team.

### Full Dream Bank
The unlabelled-data analysis was possible thanks to the [Dream Bank (and pre-scraped) data scraper](https://github.com/mattbierner/DreamScrape) from Matt Bierner.
