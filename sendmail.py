import smtplib
import pandas as pd
import db_connect
from tabulate import tabulate

server = smtplib.SMTP("smtp.gmail.com", 587)


def get_data(email):
    # data = pd.read_csv("collected_data.csv",usecols = ["email_id","phrase_text","predicted_class","user_class"])
    data = db_connect.get_data_for_email(email)
    if data.empty:
        return "NO DATA FOUND"
    else:
        data.drop_duplicates(keep='first', inplace=True)
        mail_data = data[["phrase", "predicted"]]
        table_data = tabulate(mail_data, headers='keys', tablefmt='rst')
        return table_data


def send_mail(name, receiver_email):

    server.connect()
    server.starttls()
    server.login("plannetsab@gmail.com", "sabapathy@123")
    subject = "Training Summary - Smart ATM"

    text = f"""Dear {name},

Thank you for your interest in training our model.
We would like to appreciate your effort. Below are your inputs to our training model.

{get_data(receiver_email)}

Please visit us back for training our model more.

Regards,
Plannet SAB Team"""

    message = 'Subject: {}\n\n{}'.format(subject, text)
    server.sendmail("plannetsab@gmail.com", to_addrs=receiver_email, msg=message)
    server.quit()
