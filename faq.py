from pandas import read_csv
import re,h5py
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, render_template
from nltk.corpus import stopwords
from stemming.porter2 import stem
from nltk.stem import WordNetLemmatizer
stop = stopwords.words('english')

import tensorflow as tf
import tensorflow_hub as hub

'''
All the query input will be converted to present tense using the lemmatiser
'''

def cleanQ(x):
    x=re.sub('[^a-zA-Z]+', ' ', x)
    x=str.lower(x)
    x=x.replace('myHR','my hr')
    x=x.replace('flexidollars','flexi dollars')
    x=' '.join([i for i in x.split() if i not in stop])
   # x=stem(x) 
    x=' '.join([WordNetLemmatizer().lemmatize(i,'v') for i in x.split()])
    return x

#### load doc embedding stored as hdf5 data structure
def read_op():
    global doc_embeddings
    with h5py.File('presentT_question_embeddings.h5', 'r') as hf:
        doc_embeddings = hf.get('presentT_question_embeddings').value
    print('loaded vectors')
read_op()

df=read_csv('new_ques.csv')

'''
Define the computation graph
'''
g = tf.Graph()
with g.as_default():
    tf.logging.set_verbosity(tf.logging.ERROR)
    text_input = tf.placeholder(dtype=tf.string, shape=[None])
    embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
    my_result = embed(text_input)
    init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
g.finalize()

# Create session and initialize.
session = tf.Session(graph=g)
session.run(init_op)


app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('ques_form.html')


@app.route('/', methods=['POST','GET'])

def my_form_post():
    # extract=[]
    ### gets the query text to this variable 
    text = request.form['text']
    print(text)
    print(type(text))
#     a,b=simialr_q(text)
    text=cleanQ(text)
    query_embeddings = session.run(my_result, feed_dict={text_input: [text]})
    sim=cosine_similarity(doc_embeddings,query_embeddings)
    ind = sim.reshape(-1).argsort(-1)[-4:]
    a,b=df['Question'][ind].values, df['Answer'][ind].values
#     output=dict(zip(a,b))
#     response1=a[3]+'---- '+b[3]
#     response2=a[2]+'----'+b[2]
#     response3=a[1]+'----'+b[1]
    
    response1=a[3]
    response2=a[2]
    response3=a[1]

    return 'question 1 ---  '+ response1+' \n '+ 'question 2 ---' + response2+'  \n  '+'question 3 ---  '+ response3

### This runs on your local host 

if __name__=='__main__':
    app.run(host='0.0.0.0',port=5000,debug=True)

