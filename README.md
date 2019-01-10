# faq-bot

####There are 3 kinds of hdf5 files which are created by the initial set of question and answers('new_ques.csv') depending on the kind of pre processing.
####The faq.py uses the stem_question_embedding.h5 as the latest version as it handles the tenses well.

####Run the faq.py . On your local go to 0.0.0.0:5000 to input the query in an html page.

####Next step:
####Create an api service where the input will be a query and return a json structure of best top 3/5 question and answer set.
#### Ex: {'q1':'a1', 'q2':'a2'}