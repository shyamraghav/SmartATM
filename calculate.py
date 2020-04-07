import pandas as pd
import db_connect


def metrics():

    # data = pd.read_csv(data_file)
    data = db_connect.get_data_for_metrics()

    data.drop_duplicates(keep='first', inplace=True)

    correct_predictions = data[data['pred'] == data['actual']].shape[0]
    total_predictions = data.shape[0]

    try:
        accuracy = correct_predictions/total_predictions
    except ZeroDivisionError:
        accuracy = 0.0    
    
    wd_count = data[data['actual'] == "WD"]["actual"].count()
    dp_count = data[data['actual'] == "DP"]["actual"].count()

    return { "accuracy" : accuracy , "train_data" : {"WD":float(wd_count),"DP":float(dp_count)}}


if __name__ == "__main__":
    metrics()
