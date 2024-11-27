import random, itertools, os, xmltodict, json, pickle, argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from nltk.tokenize import word_tokenize

def get_general_emotions(emot_dict):
    if emot_dict == {}:
        return 'Missing'
    else:
        EMOTIONS = []
        for chrct, emtns in emot_dict.items():
            EMOTIONS.extend(emtns)

        return ('_').join(EMOTIONS)

def get_dreamer_emotions(emot_dict):
    EMOTIONS = []

    for chrct, emtns in emot_dict.items():
        if chrct == 'D':
            EMOTIONS.extend(emtns)
        else:
            continue

    if EMOTIONS == []:
        return 'Missing'
    else:
        return ('_').join(EMOTIONS)


def get_general_emotions(emot_dict):
    if emot_dict == {}:
        return 'Missing'
    else:
        EMOTIONS = []
        for chrct, emtns in emot_dict.items():
            EMOTIONS.extend(emtns)

        return ('_').join(EMOTIONS)

def get_Emotions(file="coded_dreams.xml"):
    tree = ET.parse(file)
    root = tree.getroot()

    lst = []

    for collection in tqdm(root):

        gender = collection.findtext("sex")
        age    = collection.findtext("age")
        typ    = collection.findtext("type")
        name   = collection.findtext("name")
        idd    = collection.findtext("id")
        time   = collection.findtext("time")

        for dream in collection.findall("dream"):
            date   = dream.findtext("date")
            date   =  date if date != None else "Missing"
            number = dream.findtext("number")
            report = dream.findtext("report")

            try:
                n_wrds = len(word_tokenize(report))
            except:
                n_wrds = 0

            Char_Emot = {}
            for emot in dream.find("codings").findall("emot"):
                E   = emot[0].text
                Chr = emot[1].text
                lcl_emot_lst = Char_Emot.get(Chr, [])
                lcl_emot_lst.append(E)
                Char_Emot[Chr] = lcl_emot_lst

            lst.append(
                    [
                    gender, age, typ, name, idd, time, 
                    date, number, report, n_wrds, 
                    Char_Emot
                    ]
            )

    return lst


parser = argparse.ArgumentParser()

parser.add_argument(
    "--file_path",
    type=str,
    default="../data/dream/coded_dreams.xml",
    help="path to the xlm file containing dreambk annotated reports."
)

args = parser.parse_args()

emot_list = get_Emotions(
    file=args.file_path
)

dreamer_emotions = []
general_emotions = []

for gender, age, typ, name, idd, time, date, number, report, n_wrds, Char_Emot in tqdm(emot_list):
    dreamer_emotions.append(get_dreamer_emotions(Char_Emot))
    general_emotions.append(get_general_emotions(Char_Emot))

df = pd.DataFrame(
    emot_list,
    columns=['gender','age','type','collection','id','time','date','number','report','# words', 'emot_dict']
)

df['General Emotions'] = general_emotions
df['Dreamer Emotions'] = dreamer_emotions

df.to_csv('Reports_with_Dreamer_and_General_Emotions_test_.csv', index=False)
