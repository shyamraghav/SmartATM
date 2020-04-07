#!/usr/bin/env python
# coding: utf-8
 
import pandas as pd
from  sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import logging
import db_connect



# Preparing the Logger
log_format = "%(levelname)s :: %(asctime)s :: %(message)s"
logging.basicConfig(filename="model_build_log.txt",level=logging.DEBUG,format=log_format)
log = logging.getLogger()



def train_model():    

    # Preparing the Logger
    log_format = "%(levelname)s :: %(asctime)s :: %(message)s"
    logging.basicConfig(filename="model_build_log.txt",level=logging.DEBUG,format=log_format)
    log = logging.getLogger()
    log.info("Session Opened") 

    # Loading to data from Database

    # +++++++++++ Changing the course from csv to MongoDB ++++++++++++ 

    # raw_data = pd.read_csv(data_file)
    raw_data = db_connect.get_data_for_training()

    log.info("Imported data from Database successfully")

    # pre-processing the data using TF-IDF vectorizer

    # def vectorize(input_data):
    # from  sklearn.feature_extraction.text import TfidfVectorizer
    vect = TfidfVectorizer(lowercase=True,stop_words=None)
    # vect_data = vect.fit_transform(input_data)
    vect_data = vect.fit_transform(raw_data['phrase'])
        
    # return vect_data

    log.info("Vectoried data completed successfully")
    pickle.dump(vect,open("use_vectorizer.pkl","wb"))

    log.info("Successfully Dumped Vectorizer. User the new vecorizer")

    # Preparing the featrure Array

    # X = vectorize(raw_data['phrase']).toarray()
    X = vect_data.toarray()

    

    log.info("Feature Preparation Complete")

    # Preparing the Y data

    le = LabelEncoder()
    Y = le.fit_transform(raw_data['class'])
    pickle.dump(le,open("use_decoder.pkl","wb"))
    log.info("Successfully Dumped Label Encoder. Use the Latest Encoder")
    log.info("Y Label preparation complete")

    # Splitting the data for training and test groups
    test_size = 0.3
    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = test_size,random_state=40)

    log.info(f"Split data complete with test size : {test_size}")

    # Initiallizing the Machine Learning Model and Training the model
    log.info("Training the ML Gaussian model")
    model = GaussianNB()
    model.fit(x_train,y_train)
    log.info("Training complete and started Predicting")

    # Prediciting the class for the test data

    y_pred = model.predict(x_test)

    # Evaluating the Model with the metrics

    score = accuracy_score(y_test,y_pred)
    # print(score)
    mat = confusion_matrix(y_test,y_pred)
    # print(mat)
    log.info(f"Prediction complete. Accuracy = {score}. and Confusion matric = {mat}")

    if(score > 0.20):
        pickle.dump(model,open('use_model.pkl','wb'))
        log.info("Dumped Model into Modle.pkl")
        log.info("Model Build Complete")  
    else:
        log.info("Could not Dump model as accuracy is less. Accuracy = {score}")
        log.info("Model Build Inomplete")

    log.info("Session Closed")

if __name__== "__main__":
    train_model()
