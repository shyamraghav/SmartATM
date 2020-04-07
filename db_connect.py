from pymongo import MongoClient
import pandas as pd

client = MongoClient("mongodb+srv://admin:admin@cluster0-ukkmd.mongodb.net/test?retryWrites=true&w=majority")

db = client["Smart_ATM"]


# Into the Database operations


def add_trainer(name, email):
    user_details = db["user_details"]
    res = list(user_details.find({"email": email}))
    user = dict(res[0])

    if len(res) == 0:
        doc_id = user_details.count_documents({})
        user_details.insert_one({"_id": "Trainer_" + str(doc_id + 1), "name": name, "email": email})
        return {"status": "user_add_successfull", "name": name, "email": email}
    else:
        if (user["name"] == name and user["email"] == email):
            return {"status": "known_user"}
        else:
            return {"status": "user_already_exit", "name": user["name"], "email": user["email"]}


def add_train_data(email, phrase, pred, actual):
    training_data = db["training_data"]

    result = training_data.find_one({"_id": email})

    if (result == None):

        training_data.insert_one({"_id": email, "phrase": [phrase], "predicted": [pred], "actual": [actual]})
        return {"status": "Train data add successfull"}


    else:
        training_data.update(
            {"_id": email},
            {"$push": {"phrase": phrase, "predicted": pred, "actual": actual}})
        return {"status": "Train data add successfull"}


def add_phrase_data(phrase, actual):
    phrase_data = db["phrase_data"]
    train_id = phrase_data.count_documents({})

    phrase_data.insert_one({"_id": "Data_" + str(train_id + 1), "phrase": phrase, "class": actual})


# Get from Database Operations

def get_data_for_email(email):
    email_data_collections = db["training_data"]
    this_email_data = email_data_collections.find({"_id": email})

    if email_data_collections.count_documents({"_id": email}) > 0:
        this_email_data = email_data_collections.find({"_id": email})
        email_data = list(this_email_data)[0]
        email_df = pd.DataFrame(email_data)
        email_df.drop("_id", axis=1, inplace=True)
        email_df.reset_index(inplace=True)
        return email_df
    else:
        return


def get_data_for_training():
    train_data_collections = db["phrase_data"]
    all_data = train_data_collections.find({})
    data_list = list(all_data)
    phrase_list = []
    class_list = []

    for each_data in data_list:
        # print(type(each_data))
        phrase_list.append(each_data["phrase"])
        class_list.append(each_data["class"])

    dict_data = {"phrase": phrase_list, "class": class_list}

    train_df = pd.DataFrame(dict_data)

    return train_df


def get_data_for_metrics():
    collections = db["training_data"]
    result = collections.find({})
    metrics_data = list(result)

    phrase = []
    pred = []
    actual = []

    for each_data in metrics_data:
        phrase.extend(each_data["phrase"])
        pred.extend(each_data["predicted"])
        actual.extend(each_data["actual"])

    dict_data = {"phrase": phrase, "pred": pred, "actual": actual}
    metrics_df = pd.DataFrame(dict_data)

    return metrics_df


if __name__ == "__main__":
    get_data_for_email("email_id")
