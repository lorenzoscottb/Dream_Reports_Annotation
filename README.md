# [Automatic Annotation of Dream Reports' Emotional Content with Large Language Models](https://aclanthology.org/2024.clpsych-1.7/)

This repository contains the code, results, and analysis of the paper, to automatically annotate dream reports' emotional content. The annotation process is largely based on pre-trained large language models (LLMs), implemented via Hugging Face 🤗. 

Since you are here, you'd likely like to check out [`DReAMy`](https://github.com/lorenzoscottb/DReAMy) 😴📝🤖!

# Content
## Experiments

The work revolves around two main sets of experiments:

- Unsupervised sentiment analysis
- Supervised text classification

Code for each set of experiments can be found in the dedicated folders inside the `Experiments` folders.

## Analysis

The analysis of the collected results can be found in the dedicated Jupiter notebooks.

## Results 

The results and predictions collected in all the experiments. 

# Trained Models
The following sections explain how to use the different deployed models. You can either follow the instructions or jump [here](https://github.com/lorenzoscottb/Dream_Reports_Annotation/blob/main/Analysis/trained_model_testing.ipynb) to see a use-case of both with Jupiter notebook.

## Main model
A link to download the weights of the main model can be found in the `Experiments/Supervised_Learning` section, together with the code used to tune the model, collect the predictions presented in the paper, as well as a basic usage example.

### Download and usage 
You can find a use-case example of the main model, together with the link to download the weights [here](https://github.com/lorenzoscottb/Dream_Reports_Annotation/tree/main/Experiments/Supervised_Learning)

## [DReAMy](https://github.com/lorenzoscottb/DReAMy) and Secondary 🤗 models 
While the main model achieves the best and most stable results, it is based on a custom architecture. Hence, setting up a classification pipeline requires more coding and dependencies. For this reason, together with the main deployed model, we trained and open-sourced a few more models, which are two LLMs "simply" tuned as `multi-label` classifiers, and solely using the the standard 🤗 trainer pipeline. Despite achieving a (slightly) lower performance, these models possess other desirable features. They can be downloaded and used via the 🤗 ```transformers``` library. Moreover, one can annotate dreams in 94 languages, while the other is based on a (English-only) "small" LLM encoder, hence requiring significantly less computational power. 

### Usage
These (and more) models (and functionality) can be directly used via [`DReAMy`](https://github.com/lorenzoscottb/DReAMy), the first NLP and AI-based Python library to analyse dream reports. See the code below for a usage example of DReAMy.

```py
import dreamy 

# get some dreams to classify
test_dreams = [
    "In my dream, I was followed by the scary monster.",
    "I was walking in a forest, surrounded by singing birds. I was calm and at peace."
]

# Setup mode and classification function
classification_type = "presence"
model_type          = "base-en"
return_type         = "distribution" # set "present" for above-threshold only
device              = "cpu"

predictions = dreamy.predict_emotions(
    dream_as_list, 
    classification_type, 
    model_type,
    return_type=return_type, 
    device=device,
)

predictions

# print the probability associated with each Hall & Van de Castle emotion:
# anger (AN) apprehension (AP), sadness (SD), confusion (CO), happiness (HA)
predictions
```
```
[[{'label': 'AN', 'score': 0.08541450649499893},
  {'label': 'AP', 'score': 0.1043919175863266},
  {'label': 'SD', 'score': 0.029732409864664078},
  {'label': 'CO', 'score': 0.18161173164844513},
  {'label': 'HA', 'score': 0.30588334798812866}],
 [{'label': 'AN', 'score': 0.11174352467060089},
  {'label': 'AP', 'score': 0.17271170020103455},
  {'label': 'SD', 'score': 0.026576947420835495},
  {'label': 'CO', 'score': 0.1214553639292717},
  {'label': 'HA', 'score': 0.22257845103740692}]
````
### Query via Spaces

To get an idea of the tasks and models available via DReAMy, you can also directly query them via the [Hugging Face Space](https://huggingface.co/spaces/DReAMy-lib/dream).

# Requirments

If you want to re-run the code of any of our experiments, please make sure to check the `*_requirements.txt` files for each set of experiments.

# Acknowledgements

## Data
### Labelled Dream Reports
As specified in the paper, the labelled data is freely available for consultation via the Dream Bank website. The labelled data adopted in the supervised experiment consists of an xlm version of Dream Bank available upon request to the Dream Bank team.

### Full Dream Bank
The unlabelled-data analysis was possible thanks to the [Dream Bank (and pre-scraped) data scraper](https://github.com/mattbierner/DreamScrape) from Matt Bierner.
