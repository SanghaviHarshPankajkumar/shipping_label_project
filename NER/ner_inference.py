
from flair.data import Sentence
from flair.models import SequenceTagger
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
import json
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
    label_name = {'GCNUM':[],'TRACK-ID':[],'Company':"" }
    for label in res['entities']:
        sorted_labels = sorted(label['labels'], key=lambda k: k['confidence']) 
        if sorted_labels[0]['value'] not in label_name:
            label_name[sorted_labels[0]['value']].append([label['text']])
        else:
            label_name[sorted_labels[0]['value']].append([label['text']])

    print(label_name)
    if label_name['GCNUM'] is None:
        pattern = re.compile(r"\b{}\b".format("[G,O,Q][C,O,Q][0-9]{5}"))
        result = pattern.search(sent)
        if result is not None:
            label_name['GCNUM'].append(result.group(0))
    
    label_name['GCNUM'] = [json.dumps(label_name['GCNUM'])]
    label_name['TRACK-ID'] = [json.dumps(label_name['TRACK-ID'])]
    #getting companie name
    #1. get name directly from sentence
    label_name['Company'] = get_company_name(sent.lower())

    #2. track id from sentence

    return label_name

def get_company_name(sent):
    patterns = []
    patterns.append({'company': "FeDex", 'pattern' : re.compile(r"\b{}\b".format("fedex")),"track_pattern":re.compile(r"\b{}\b".format("0003315500"))})
    patterns.append({'company': "FeDex", 'pattern' : re.compile(r"\b{}\b".format("fedex")),"track_pattern":re.compile(r"\b{}\b".format("[0-9]{12}"))})
    patterns.append({'company':'UPS', 'pattern':re.compile(r"\b{}\b".format("ups")),"track_pattern":re.compile(r"\b{}\b".format("1Z"))})
    patterns.append({'company':'USPS', 'pattern':re.compile(r"\b{}\b".format("usps")),"track_pattern":re.compile(r"\b{}\b".format("42033155"))})
    patterns.append({'company':'onTrack', 'pattern':re.compile(r"\b{}\b".format("ontrack")),"track_pattern":re.compile(r"\b{}\b".format("BG[0-9]{5}"))})
    patterns.append({'company':'Lasership', 'pattern':re.compile(r"\b{}\b".format('1LS')),"track_pattern":re.compile(r"\b{}\b".format('1LS'))})
    patterns.append({'company':'Amazon', 'pattern':re.compile(r"\b{}\b".format("dmi6 | dm16")),"track_pattern":re.compile(r"\b{}\b".format("T[B,8]A"))})

    for pattern in patterns:
        if pattern['pattern'].search(sent) is not None:
            return pattern['company']
    
    for pattern in patterns:
        if pattern['track_pattern'].search(sent) is not None:
            return pattern['company']
    return None


