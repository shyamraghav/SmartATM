import pandas as pd
from  sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import logging

import train

model = pickle.load(open("use_model.pkl","rb"))
vect = pickle.load(open("use_vectorizer.pkl","rb"))
le = pickle.load(open("use_decoder.pkl","rb"))



def predict_class(phrase):

    v_que = vect.transform([phrase])
    out = model.predict(v_que.toarray())
    cls =  le.inverse_transform(out)
    if (cls[0] == "WD"):
        return "Withdrawal"
    else:
        return "Deposit"    
    
