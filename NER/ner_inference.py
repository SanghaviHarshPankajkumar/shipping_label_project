
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

    return label_name