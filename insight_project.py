
import torch
import transformers as ppb
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.nn import CrossEntropyLoss, MSELoss
import streamlit as st
from tqdm import tqdm_notebook, trange
import os
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

import pickle

#First trying DistilBERT
model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

## Want BERT instead of distilBERT? Uncomment the following line:
#model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')

# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

st.title('Streamline positive black news')
st.write('Streamlining relevant, balanced Black news articles to the news feed')


st.write('Upload a csv file with news headlines in one column and choose a news category')
uploaded_file = st.sidebar.file_uploader("Upload News Headlines", type=['csv'])

if uploaded_file is not None:
    x = pd.read_csv(uploaded_file)
    x = x.iloc[:,0].values
else:
    print('Upload News Article')

sugcat = add_selectbox = st.sidebar.selectbox(
    'Choose a news category',
    ('Business', 'General', 'Sports'))


st.write('Articles for the', sugcat, 'category')



#PROGRESS BAR



#running cat classifier
trial = pd.Series(x)

filenamec = 'badcat_class.sav'

cat_model = pickle.load(open(filenamec, 'rb'))


ptokenized = trial.apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
max_len = 0
for i in ptokenized.values:
    if len(i) > max_len:
        max_len = len(i)

ppadded = np.array([i + [0]*(max_len-len(i)) for i in ptokenized.values])
pattention_mask = np.where(np.array(ppadded)!=0,1,0)
pinput_ids = torch.tensor(ppadded)
pattention_mask = torch.tensor(pattention_mask)
pinput_ids = pinput_ids.clone().detach().to(torch.int64)

with torch.no_grad():
    print('...')
    last_hidden_states = model(pinput_ids, attention_mask=pattention_mask)

features = last_hidden_states[0][:,0,:].numpy()

predc = pd.Series(cat_model.predict(features), name = 'category')

output = pd.concat([predc, trial], axis=1)

if sugcat == 'Business':
    cat_news = output[output.category == '0']
elif sugcat == 'Sports':
    cat_news = output[output.category == '2']
else:
    cat_news = output[output.category == '1']

newwords = cat_news.iloc[:,0]


ptokenized = newwords.apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
max_len = 0
for i in ptokenized.values:
    if len(i) > max_len:
        max_len = len(i)

ppadded = np.array([i + [0]*(max_len-len(i)) for i in ptokenized.values])
pattention_mask = np.where(np.array(ppadded)!=0,1,0)
pinput_ids = torch.tensor(ppadded)
pattention_mask = torch.tensor(pattention_mask)
pinput_ids = pinput_ids.clone().detach().to(torch.int64)

with torch.no_grad():
    print('...')
    last_hidden_states = model(pinput_ids, attention_mask=pattention_mask)
features = last_hidden_states[0][:,0,:].numpy()


filename = 'rfclf.sav'
top_model = pickle.load(open(filename, 'rb'))
pred = pd.Series(top_model.predict(features), name = 'sentiment')

output = pd.concat([pred, trial], axis=1)
pos = output[output.sentiment == '2']
neg = output[output.sentiment== '0']
neut = output[output.sentiment== '1']

if not pos.empty:
    st.write('Positive articles:')
    st.write(pos.iloc[:,1].values)
if not neg.empty:
    st.write('Negative articles:')
    st.write(neg.iloc[:,1].values)
if not neut.empty:
    st.write('Neutral articles:')
    st.write(neut.iloc[:,1].values)
