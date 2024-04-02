
from flair.data import Sentence
from flair.models import SequenceTagger
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

import os
from dotenv import load_dotenv
from pathlib import Path
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)
import re

path = {
        'NER_MODEL_PATH': str(os.getenv('NER_MODEL_PATH')),
       }
model = SequenceTagger.load(path['NER_MODEL_PATH'])


def inference(sent):
    # print("sent is get",sent)
    sentence = Sentence(sent)
    res = model.predict(sentence)

    res = sentence.to_dict()
    label_name = {}
    for label in res['entities']:
        sorted_labels = sorted(label['labels'], key=lambda k: k['confidence']) 
        if sorted_labels[0]['value'] not in label_name:
            label_name[sorted_labels[0]['value']] = [label['text']]
        else:
            label_name[sorted_labels[0]['value']].append(label['text'])


    #getting companie name
    #1. get name directly from sentence
    # company_name = get_company_name(sent.lower())
    #2. track id from sentence

    return label_name

def get_company_name(sent):
    patterns = []
    patterns.append({'company': "FeDex", 'pattern' : re.compile(r"\b{}\b".format("fedex")),"track_pattern":re.compile(r"\b{}\b".format("0003315500"))})
    patterns.append({'company':'UPS', 'pattern':re.compile(r"\b{}\b".format("ups")),"track_pattern":re.compile(r"\b{}\b".format("1Z"))})
    patterns.append({'company':'USPS', 'pattern':re.compile(r"\b{}\b".format("usps")),"track_pattern":re.compile(r"\b{}\b".format("42033155"))})
    patterns.append({'company':'onTrack', 'pattern':re.compile(r"\b{}\b".format("ontrack")),"track_pattern":re.compile(r"\b{}\b".format("[0-9]{12}"))})
    patterns.append({'company':'Lasership', 'pattern':re.compile(r"\b{}\b".format('1LS')),"track_pattern":re.compile(r"\b{}\b".format('1LS'))})
    patterns.append({'company':'Amazon', 'pattern':re.compile(r"\b{}\b".format("DM16")),"track_pattern":re.compile(r"\b{}\b".format("TBA"))})

   
    for pattern in patterns:
        if pattern['pattern'].search(sent) is not None:
            return pattern['company']
    return None


