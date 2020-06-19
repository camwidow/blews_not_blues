
#import torch
#import transformers as ppb
#from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
#from torch.nn import CrossEntropyLoss, MSELoss
import streamlit as st
from tqdm import tqdm_notebook, trange
import os
#from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification
#from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
import time
import pickle
import inflect

@st.cache(allow_output_mutation = True)
def  load_cat():
    cat_model = pickle.load(open('cat_cheap_class.sav', 'rb'))
    return cat_model
cat_model = load_cat()

@st.cache(allow_output_mutation=True)
def load_sent():
    sent_model = pickle.load(open('cheap_class.sav', 'rb'))
    return sent_model
sent_model = load_sent()

@st.cache(allow_output_mutation=True)
def load_tf1():
    tf1 = pickle.load(open('VECTORIZER.pkl', 'rb'))
    return tf1
tf1 = load_tf1()


# Create new tfidfVectorizer with old vocabulary

tf1_new = TfidfVectorizer(analyzer='word', stop_words = "english", lowercase = True,
                           vocabulary = tf1.vocabulary_)
#
#First trying DistilBERT
#model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

## Want BERT instead of distilBERT? Uncomment the following line:
#model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
# Load pretrained model/tokenizer
#tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
#model = model_class.from_pretrained(pretrained_weights)
#tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
#model = model_class.from_pretrained(pretrained_weights)

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()
p = inflect.engine()

@st.cache
def converttostr(input_seq, seperator):
    final_str = seperator.join(input_seq)
    return final_str
@st.cache
def cv(data):

    count_vectorizer = CountVectorizer()

    emp = count_vectorizer.fit_transform(data)

    return emp, count_vectorizer
@st.cache
def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]
@st.cache
def standardize_text(df, text_field):
    """ Function for cleaning text"""
    df[text_field] = df[text_field].str.replace(r"http\S+", "")
    df[text_field] = df[text_field].str.replace(r"http", "")
    df[text_field] = df[text_field].str.replace(r"@\S+", "")
    df[text_field] = df[text_field].str.replace(r"[^A-Za-z0-9()!?@\'\`\"\_\n]", " ")
    df[text_field] = df[text_field].str.replace(r"/", " ")
    df[text_field] = df[text_field].str.replace(r"''", " ")
    df[text_field] = df[text_field].str.replace(r",", " ")
    df[text_field] = df[text_field].str.replace(r"@", "at")
    df[text_field] = df[text_field].str.replace(r"!", "")
    df[text_field] = df[text_field].str.lower()
    return df
@st.cache
def convert_number(text):
    # split string into list of words
    temp_str = text.split()
    # initialise empty list
    new_string = []

    for word in temp_str:
        # if word is a digit, convert the digit
        # to numbers and append into the new_string list
        if word.isdigit():
            temp = p.number_to_words(word)
            new_string.append(temp)
  # append the word as it is
        else:
            new_string.append(word)
# join the words of new_string to form a string
    temp_str = ' '.join(new_string)
    return temp_str


st.title('Good Blews, not Blues')

st.header('Streamlining relevant, positive, Black news headlines')

st.subheader("Enter a headline below and click 'classify' predict its category (General,Sports, Business) and sentiment score!")

st.write('For example:')

titles = ['Walmart Says It Will No Longer Lock Up African-American Beauty Products',"13 Investigates: 'No need for disparity' in African American arrest rates","Top Tulsa police officer: 'We're shooting African Americans about 24 percent less than we probably ought to be",'Timberwolves president Gersson Rosas hopeful for more front-office diversity in pro sports',
'African American owned businesses hurt most by the lockdowns',
'Local Entrepreneurs Aim To Create African American Business Listing']
dff =pd.DataFrame(data = titles, columns = ['Recent Black headlines from June 10th'])
#st.write(dff)
st.table(dff)

if st.button('Click button to generate sentiment score!'):
    #PREPROCESSING

    x =standardize_text(dff, 'Recent Black headlines from June 10th') #STRING
    x = x['Recent Black headlines from June 10th'].apply(convert_number)
    x = list(x)
    X_tf1 = tf1_new.fit_transform(x)
    pred = cat_model.predict(X_tf1)
    st.write('Are predicted as such:')
    predictions = []
    for i in pred:
        if i == 0:
            statement = ('Category: Business')
        elif i == 1:
            statement = ('Category: General')
        elif i == 2:
            statement = ('Category: Sports')
        predictions.append(statement)

    pred = sent_model.predict(X_tf1)

    sents = []


    for i in pred:
        if i == 0:
            sent = ('Sentiment: Negative')
        elif i == 1:
            sent = ('Sentiment: Neutral')
        elif i == 2:
            sent = ('Sentiment: Positive')
        sents.append(sent)


    for i in range(len(pred)):
        j = dff.values
        st.write(str(j[i]))
        st.write(predictions[i],'|',sents[i])
        st.write('---------------')





        #X_tf1 = tf1_new.fit_transform(x)






st.header('Now you try!')
st.write('Type an article headline and click "classify"!')

#st.write('Upload a csv file with news headlines in one column and choose a news category')
#uploaded_file = st.sidebar.file_uploader("Upload News Headlines", type=['csv'])

# if uploaded_file is not None:
#     x = pd.read_csv(uploaded_file)
#     x = x.iloc[:,0].values
# else:
#     print('Upload News Article')



# sugcat = add_selectbox = st.sidebar.selectbox(
#     'Choose a news category',
#     ('Business', 'General', 'Sports'))


#st.write('Articles for the', sugcat, 'category')



#PROGRESS BAR

x = st.text_input('Article Title')


if st.button('classify'):

    if x == "":
        st.subheader('Insert Article title and then click "classify"')
        #break
    else:

        titles = [x]

        dff =pd.DataFrame(data = titles, columns = ['Recent Black headlines from June 10th'])


    #PREPROCESSING
        x =standardize_text(dff, 'Recent Black headlines from June 10th') #STRING

        x = x['Recent Black headlines from June 10th'].apply(convert_number)
        x = list(x)
        X_tf1 = tf1_new.fit_transform(x)
        pred = cat_model.predict(X_tf1)
        predictions = []
        for i in pred:
            if i == 0:
                statement = ('Category: Business')
            elif i == 1:
                statement = ('Category: General')
            elif i == 2:
                statement = ('Category: Sports')
            predictions.append(statement)


        pred = sent_model.predict(X_tf1)

        sents = []

        for i in pred:
            if i == 0:
                sent = ('Sentiment: Negative')
            elif i == 1:
                sent = ('Sentiment: Neutral')
            elif i == 2:
                sent = ('Sentiment: Positive')
                sents.append(sent)

        for i in range(len(pred)):
            st.write(dff.iloc[i,:],predictions[i],'|',sents[i])























    #
    #
    # df = pd.DataFrame([x],columns = ['title'])
    # x =standardize_text(df, 'title')
    # x = convert_number(x.title[0])
    # x = lemmatize_text(x)
    # filenamec = 'cat_cheap_class.sav'
    # cat_model = pickle.load(open(filenamec, 'rb'))
    # tfidf = pickle.load(open("VECTORIZER.pkl", 'rb'))
    # tf1_new = TfidfVectorizer(analyzer='word', stop_words = "english", lowercase = True, vocabulary = tfidf.vocabulary_)
    # X_tf1 = tf1_new.fit_transform(x)
    # pred = cat_model.predict(X_tf1)
    #
    # if 0 in pred:
    #     st.write('Category: NEGATIVE')
    # elif 1 in pred:
    #     st.write('Sentiment: POSITIVE')
    # elif 2 in pred:
    #     st.write('Sentiment: NEUTRAL')
    #


    #
    #
    #
    #
    #
    #
    # ptokenized = trial.apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
    # max_len = 0
    # for i in ptokenized.values:
    #     if len(i) > max_len:
    #         max_len = len(i)
    # ppadded = np.array([i + [0]*(max_len-len(i)) for i in ptokenized.values])
    # pattention_mask = np.where(np.array(ppadded)!=0,1,0)
    # pinput_ids = torch.tensor(ppadded)
    # pattention_mask = torch.tensor(pattention_mask)
    # pinput_ids = pinput_ids.clone().detach().to(torch.int64)
    #
    # with torch.no_grad():
    #     last_hidden_states = model(pinput_ids, attention_mask=pattention_mask)
    # features = last_hidden_states[0][:,0,:].numpy()
    #
    # predc = cat_model.predict(features)
    # if predc == '0':
    #     st.write('Category: Business')
    # elif predc == '2':
    #     st.write('Category: Sports')
    # else:
    #     st.write('Category: General')
    # ptokenized = trial.apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
    # max_len = 0
    # for i in ptokenized.values:
    #     if len(i) > max_len:
    #         max_len = len(i)
    # ppadded = np.array([i + [0]*(max_len-len(i)) for i in ptokenized.values])
    # pattention_mask = np.where(np.array(ppadded)!=0,1,0)
    # pinput_ids = torch.tensor(ppadded)
    # pattention_mask = torch.tensor(pattention_mask)
    # pinput_ids = pinput_ids.clone().detach().to(torch.int64)
    # with torch.no_grad():
    #     last_hidden_states = model(pinput_ids, attention_mask=pattention_mask)
    # features = last_hidden_states[0][:,0,:].numpy()
    # filename = 'cheap_class.sav'
    # top_model = pickle.load(open(filename, 'rb'))
    # pred = top_model.predict(features)
    # st.write("''",x,"''",'is a:')
    # if pred == '0':
    #     st.write('Negative Article')
    # elif pred == '2':
    #     st.write('Positive Article')
    # else:
    #     st.write('Neutral Article')
