from flask import Flask, jsonify
from flask_cors import CORS
import train
import predict
import calculate
import json
import logging
import sendmail
import db_connect

msg_format = "%(levelname)s :: %(asctime)s :: %(message)s"
logging.basicConfig(filename="API_log.txt", level=logging.DEBUG, format=msg_format)
apilog = logging.getLogger()
# apilog.info("API Running")

app = Flask(__name__)
CORS(app)


@app.route('/smartatm/<phrase>')
def smartatm(phrase):
    result = predict.predict_class(phrase)
    apilog.info("Predict Request : " + phrase)
    apilog.info("Predicted as    : " + result)
    return result
   

@app.route('/sendmail/<name>/<email>')    
def send_mail(name, email):
    sendmail.send_mail(name, email)
    return "ok"


@app.route('/')
def health():
    return "ok"


@app.route('/metrics')
def graph_data():
    return calculate.metrics()


@app.route('/update_confidence/<name>/<email>/<phrase>/<pred>/<ans>')
def update_confidence(name, email, phrase, pred, ans):
    apilog.info("Evaluation for :" + phrase)
    apilog.info("Predicted as : " + pred + " , User Answer : " + ans)
    # access_file = open("collected_data.csv","a")
    # add_line = "\n" + name + ',' + email + ',' + phrase + ',' + pred + ',' + ans
    # access_file.write(add_line)
    # access_file.close()

    db_connect.add_train_data(email,phrase,pred,ans)
    accuracy = calculate.metrics()

    if pred != ans:
        # apilog.info("Initiated Retraining")
        try:

            db_connect.add_phrase_data(phrase,ans)
            
            train.train_model()

            apilog.info("Retrain Complete")
        # print("retrain initiated and complete")
        except Exception as e:
            return str(e)
    return accuracy


if __name__ == "__main__":
    app.run(debug=True)
