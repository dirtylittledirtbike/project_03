#hey
import dash
import dash_html_components as html
import dash_core_components as dcc
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from nltk import RegexpTokenizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

import plotly.express as px
from src.model_insights import get_word_covariance, get_class_features

# doing this so we can see full words
pd.set_option('display.max_colwidth', None)

with open('pickle_files/stop_words.pickle', 'rb') as file:
    stop_words_complete = pickle.load(file)

class Tokenizer(object):
    def __init__(self):
        self.pt = PorterStemmer()
        self.wnl = WordNetLemmatizer()
        self.tk = RegexpTokenizer(r'\b[a-zA-Z]{3,}\b')
        self.stpwrd = set(stop_words_complete)
    
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in self.tk.tokenize(doc) if not t in self.stpwrd]

# remove this later/put in model_insights
def get_topic_words(model, feature_names, n_top_words):
    topic_words = []
    for topic_idx, topic in enumerate(model.components_):
        message = ""
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        topic_words.append(message)
    return topic_words

# getting surrounding words from input in topics
def get_topic_matches(input_str):
    idxs = []
    for item in all_topic_words:
        if input_str in item:
            idxs.append(all_topic_words.index(item))

    matches = []
    for i in idxs:
        # finding surrounding words with re i know its sloppy but deadlines
        sub = '(\w*)\W*(\w*)\W*(\w*)\W*(\w*)\W*({})\W*(\w*)\W*(\w*)\W*(\w*)\W*(\w*)\W*(\w*)'.format(input_str)
        str1 = all_topic_words[i]
        #printing the topic we are on
        for j in re.findall(sub, str1, re.I):
            words = " ".join([x for x in j if x != ""])
            matches.append([str(i), words])

    return matches

my_tokenizer = Tokenizer()

with open('pickle_files/vectorizer.pickle', 'rb') as file:
    tfdif_vectorizer = pickle.load(file)
    X_train_vect = pickle.load(file)
    X_val_vect = pickle.load(file)
    nb = pickle.load(file)

with open('pickle_files/nmf_model.pickle', 'rb') as file:
    nmf = pickle.load(file)

tfidf_feature_names = tfdif_vectorizer.get_feature_names()
n_top_words = 500
all_topic_words = get_topic_words(nmf, tfidf_feature_names, n_top_words)
df, cov = get_word_covariance(tfdif_vectorizer, nb, n=500, top=True)

# generate table to display values
def generate_table(dataframe, max_rows=10):
    return html.Table([
                       html.Thead(
                                  html.Tr([html.Th(col) for col in dataframe.columns])
                                  ),
                       html.Tbody([
                                   html.Tr([
                                            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
                                            ]) for i in range(min(len(dataframe), max_rows))
                                   ])
                       ])
# ------------------------- dash app -------------------------

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = '🎠'


app.layout = html.Div([
                       html.Div([
                                 html.Label('Word:'),
                                 dcc.Dropdown(
                                              id='demo-dropdown', value='please',
                                              options=[
                                                       {'label': k, 'value': k} for k in df.index.to_list()
                                                       ],
                                              ),
                                
                                ],style={'width':'30%', 'height':'auto', 'display':'grid', 'width':'40%'}),
                       html.Br(),
                       html.Br(),
                       html.Div(id='dd-output-container', style = {'display':'flex'}),
                       ]
                      )


@app.callback(
              dash.dependencies.Output('dd-output-container', 'children'),
              [dash.dependencies.Input('demo-dropdown', 'value')])
def update_output(value):
    
    df = pd.DataFrame(get_topic_matches(value))
    df.columns = ['topic', 'document text']
    
    # this is going to be for visualizing the graph
    topics = 10
    cols = ['topic' + str(i) for i in range(topics)]
    topic_df = pd.DataFrame(nmf.components_, index=cols, columns=tfdif_vectorizer.get_feature_names()).T
    neg, pos = get_class_features(tfdif_vectorizer, nb, n=20, top=True)
    topic_formatted = topic_df.T[pos].T
    topic_formatted.head()
    
    
    data_canada = px.data.gapminder().query("country == 'Canada'")
    fig = px.bar(data_canada, x='year', y='pop', width=600, height=500)
    
    return generate_table(df),dcc.Graph(id='output-graph', figure=fig)


if __name__ == '__main__':
    app.run_server(debug=True)
