import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader

import re
import numpy as np
import pandas as pd
import copy
import shutil
import transformers, datasets
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.t5.modeling_t5 import T5Config, T5PreTrainedModel, T5Stack
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers import T5EncoderModel, T5Tokenizer
from transformers import TrainingArguments, Trainer, set_seed

from evaluate import load
from datasets import Dataset

from tqdm import tqdm
import random

from scipy import stats

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef

import matplotlib.pyplot as plt
import argparse



# Set random seeds for reproducibility of your trainings run
def set_seeds(s):
    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)
    set_seed(s)

# Dataset creation
def create_dataset(tokenizer,seqs,labels):
    tokenized = tokenizer(seqs, max_length=1024, padding=False, truncation=True)
    dataset = Dataset.from_dict(tokenized)
    dataset = dataset.add_column("labels", labels)

    return dataset



def load_model(filepath, num_labels=1, mixed = False):
# Creates a new PT5 model and loads the finetuned weights from a file

    # load a new model
    model, tokenizer = PT5_classification_model(num_labels=num_labels, half_precision=mixed)
    
    # Load the non-frozen parameters from the saved file
    non_frozen_params = torch.load(filepath)

    # Assign the non-frozen parameters to the corresponding parameters of the model
    for param_name, param in model.named_parameters():
        if param_name in non_frozen_params:
            param.data = non_frozen_params[param_name].data

    return tokenizer, model
def data_preprocess(datapath):
    df = pd.read_csv(datapath)
    df = df.rename(columns={'seq':'sequence'})
    df = df.reset_index(drop=True)
    df['sequence']=df.apply(lambda row : " ".join(row["sequence"]), axis = 1)
    return df[["sequence","label"]]



datapath = f"/home/data/"
test_datapath = f"{datapath}/test_data.csv" #输入数据
print("test data is: ", test_datapath)


workpath = f"/home/result"#设置结果输出地址
os.makedirs(workpath, exist_ok=True)
os.makedirs(f'{workpath}/test', exist_ok=True)
os.chdir(workpath)
print("workpath is: ", workpath)


test_df = data_preprocess(test_datapath)

tokenizer, model_reload_ori = load_model("/home/model_finetuned.pth", num_labels=1, mixed=False)
print("加载模型：/home/model_finetuned.pth")
print("tokenizer:", type(tokenizer), tokenizer is None)
print("model:", type(model_reload_ori), model_reload_ori is None)

model_reload=model_reload_ori.to("cpu")

################
'''
test
'''
################
#Use reloaded model
model = model_reload

# Set the device to use
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# create Dataset
test_set=create_dataset(tokenizer,list(test_df['sequence']),list(test_df['label']))

# make compatible with torch DataLoader
test_set = test_set.with_format("torch", device=device)

# Create a dataloader for the test dataset
test_dataloader = DataLoader(test_set, batch_size=16, shuffle=False)

# Put the model in evaluation mode
model.eval()

# Make predictions on the test dataset
predictions = []
with torch.no_grad():
    for batch in tqdm(test_dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        #add batch results(logits) to predictions
        predictions +=F.softmax(model(input_ids, attention_mask=attention_mask).logits).tolist()

prob = [row[1] for row in predictions]

with open("./test/y_true.txt","w") as f:
    f.write(str(test_df.label.to_list()))
with open("./test/y_pred.txt","w") as f:
    f.write(str(prob))
