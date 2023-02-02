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

Setup and run the classification pipeline
```py
from transformers import pipeline

# get some dream to classify
test_dreams = [
    "In my dream, I was followed by the scary monster.",
    "I was walking in a forest, surrounded by singing birds. I was calm and at peace."
]

# set up the pipeline
classifier = pipeline(
    task="text-classification", 
    model=model, 
    tokenizer=tokenizer,
    top_k=None, # set to k=n if just need top n classes instead of all
)

# get the model's classification
predictions = classifier(test_dreams)

# print the probability associated with each Hall & Van de Castle emotion:
# anger (AN) apprehension (AP), sadness (SD), confusion (CO), happiness (HA)
predictions
>>> [[{'label': 'AP', 'score': 0.8697441816329956},
>>>   {'label': 'CO', 'score': 0.1245221346616745},
>>>   {'label': 'HA', 'score': 0.025534192100167274},
>>>   {'label': 'AN', 'score': 0.015074575319886208},
>>>   {'label': 'SD', 'score': 0.010451494716107845}],
>>>  [{'label': 'HA', 'score': 0.9519748091697693},
>>>   {'label': 'AP', 'score': 0.07662183046340942},
>>>   {'label': 'SD', 'score': 0.042797815054655075},
>>>   {'label': 'CO', 'score': 0.02953989803791046},
>>>   {'label': 'AN', 'score': 0.008983743377029896}]]
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
