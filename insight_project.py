
import streamlit as st
from tqdm import tqdm_notebook, trange
import os
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
    cat_model = pickle.load(open('cat_classifier.sav', 'rb'))
    return cat_model
cat_model = load_cat()

@st.cache(allow_output_mutation=True)
def load_sent():
    sent_model = pickle.load(open('sent_classifier.sav', 'rb'))
    return sent_model
sent_model = load_sent()



st.title('Good [Bl]ews, not Blues')

st.header('Streamlining relevant, positive Black news')

st.subheader("Enter a headline below and click 'classify' to predict its category (General,Sports, Business) and sentiment!")

st.write('For example:')




titles = ['Walmart Says It Will No Longer Lock Up African-American Beauty Products',"13 Investigates: 'No need for disparity' in African American arrest rates","Top Tulsa police officer: 'We're shooting African Americans about 24 percent less than we probably ought to be",'Timberwolves president Gersson Rosas hopeful for more front-office diversity in pro sports',
'African American owned businesses hurt most by the lockdowns',
'Local Entrepreneurs Aim To Create African American Business Listing']
dff =pd.DataFrame(data = titles, columns = ['Recent Black headlines from June 10th'])
#st.write(dff)
st.table(dff)


#Example:

if st.button('Click button to generate sentiment score!'):
    #PREPROCESSING

    pred = cat_model.predict(dff['Recent Black headlines from June 10th'])
    st.write('Are predicted as such:')
    predictions = []
    for i in pred:
        if i == 0:
            statement = ('Business')
        elif i == 1:
            statement = ('General')
        elif i == 2:
            statement = ('Sports')
        predictions.append(statement)

    pred = sent_model.predict(dff['Recent Black headlines from June 10th'])

    sents = []


    for i in pred:
        if i == 0:
            sent = ('Negative')
        elif i == 1:
            sent = ('Neutral')
        elif i == 2:
            sent = ('Positive')
        sents.append(sent)

    st.write('Category | Sentiment')
    st.write('--------------------')
    for i in range(len(pred)):
        j = dff.values
        st.write(str(j[i]))
        st.write(predictions[i],'|',sents[i])
        st.write('---------------')


st.header('Now you try!')
st.write('Type an article headline and click "classify"!')

#User input

x = st.text_input('Article Title')


if st.button('classify'):

    if x == "":
        st.subheader('Insert Article title and then click "classify"')
        #break
    else:

        cat = cat_model.predict(pd.Series(x))

        predcat = []
        for i in cat:
            if i == 0:
                statement = ('Category: Sports')
            elif i == 1:
                statement = ('Category: General')
            elif i == 2:
                statement = ('Category: Business')
            predcat.append(statement)


        sent = sent_model.predict(pd.Series(x))

        predsent = []

        for i in sent:
            if i == 0:
                statement = ('Sentiment: Negative')
            elif i == 1:
                statement = ('Sentiment: Neutral')
            elif i == 2:
                statement = ('Sentiment: Positive')
            predsent.append(statement)

        j = x
        st.write(j)
        for i in range(len(predsent)):
            st.write(predcat[i],'|',predsent[i])

#CSV Upload

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
