import pandas as pd
import numpy as np
from pathlib import Path
import torch
import apex
import os
import logging

from pytorch_transformers import BertTokenizer
from fast_bert.data_cls import BertDataBunch
from fast_bert.learner_cls import BertLearner
from fast_bert.metrics import accuracy
from fast_bert.prediction import BertClassificationPredictor

# check if (multiple) GPUs are available

multi_gpu=False

if torch.cuda.is_available():
    
    device_cuda = torch.device("cuda")
    
    if torch.cuda.device_count() > 1:
        multi_gpu = True
else:
    device_cuda = torch.device("cpu")
    
print (multi_gpu)

logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger()

metrics = []
metrics.append({'name': 'accuracy', 'function': accuracy})

BASE = Path('data/readmission_prediction//')
LABEL_PATH = BASE
BIOBERT_PATH = Path('biobert/')

def train(path_to_directory, model):
        
    DATA_PATH = BASE/path_to_directory
    OUTPUT_DIR = DATA_PATH/'output'/model 
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    
    if (model == "biobert"):
        tokenizer = BertTokenizer.from_pretrained(BIOBERT_PATH, 
                                                  do_lower_case=True)
        pretrained_path=BIOBERT_PATH
        model_type = "bert"
    elif (model == "bert"):
        tokenizer = "bert-base-uncased"
        pretrained_path="bert-base-uncased"
        model_type = "bert"
    elif (model == "xlnet"):
        tokenizer = "xlnet-base-cased"
        pretrained_path="xlnet"
        model_type = "xlnet"
    else:
        print ("Model parameter must be either 'bert' or 'biobert'")
        return
    
    databunch = BertDataBunch(DATA_PATH, 
                              LABEL_PATH,
                              tokenizer=tokenizer,
                              train_file='train.csv',
                              val_file='val.csv',
                              text_col='text',
                              label_file='labels.csv',
                              label_col='30d_unplan_readmit',
                              batch_size_per_gpu=10,
                              max_seq_length=512,
                              multi_gpu=multi_gpu,
                              multi_label=False,
                              model_type=model_type)
                              #clear_cache=True)
    
    learner = BertLearner.from_pretrained_model(databunch,
                                                pretrained_path=pretrained_path,
                                                metrics=metrics,
                                                device=device_cuda,
                                                logger=logger,
                                                output_dir=OUTPUT_DIR,
                                                finetuned_wgts_path=None,
                                                warmup_steps=500,
                                                multi_gpu=multi_gpu,
                                                is_fp16=True,
                                                multi_label=False,
                                                logging_steps=40)
    
    if path_to_directory.split('/',1)[1] in ['original','synthetic']:
        epochs = 6
    else:
        epochs = 3
        
    learner.fit(epochs=epochs,
                lr=6e-5,
                validate=True, # Evaluate the model after each epoch
                schedule_type="warmup_cosine")
    
    learner.save_model()
    
    return

#for directory in ['original','original_2x','synthetic','original_eda']:
#    for model in ['biobert','bert']:
#        train('transformer/'+directory, model)
train('transformer/original_eda','biobert')
train('transformer/original_eda','bert')
