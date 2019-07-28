import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import mlflow.sklearn


def get_criterions():
    return ["A1", "B1", "B2", "B3", "B4", "B5", "C1", "C2", "D1", "D2", "D3", "D4", "D5", "D6", "D7", "E1", "E2", "E3",
     "E4", "E5", "E6", "F1", "G1"]

def prepare_data(data):
    """
        Return a prepared dataframe
        input : Dataframe with expected schema

    """
    dataframe_targets = data.groupby("transcript_id").sum()[get_criterions()]
    data_frame_text_fields = data.groupby("transcript_id")["text"].agg(lambda col: ''.join(col))
    data_frame_text_fields = data_frame_text_fields.to_frame()
    data_frame_text_fields.reset_index(level=0, inplace=True)
    dataframe_targets.reset_index(level=0, inplace=True)
    data_frame_merged = pd.merge(dataframe_targets, data_frame_text_fields, on="transcript_id")
    criterions = ["A1", "B1", "B2", "B3", "B4", "B5", "C1", "C2", "D1", "D2", "D3", "D4", "D5", "D6", "D7", "E1", "E2",
                  "E3", "E4", "E5", "E6", "F1", "G1"]

    processed_df = data_frame_merged
    for criterion in criterions:
        processed_df[criterion] = processed_df[criterion].apply(lambda x: 1 if x >= 0.5 else 0)

    return processed_df

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file (make sure you're running this from the root of MLflow!)
    wine_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/transcripts/PTSD_data.csv")
    data = pd.read_csv(wine_path)

    prepared_data = prepare_data(data)

    train, test = train_test_split(prepared_data, random_state=42, test_size=0.33, shuffle=True)
    X_train = train.text
    X_test = test.text

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run():

        LogReg_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2),
                                      stop_words='english')),
            ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=1)),
        ])

        criterions = get_criterions()

        for category in criterions:
            print('... Processing {}'.format(category))
            # train the model using X_dtm & y
            LogReg_pipeline.fit(X_train, train[category])
            # compute the testing accuracy
            prediction = LogReg_pipeline.predict(X_test)
            print('Test accuracy is {}'.format(accuracy_score(test[category], prediction)))

        #predicted_qualities = lr.predict(test_x)

        #(rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        #print("Logistic Regression model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        #print("  RMSE: %s" % rmse)
        #print("  MAE: %s" % mae)
        #print("  R2: %s" % r2)

        #mlflow.log_param("alpha", alpha)
        #mlflow.log_param("l1_ratio", l1_ratio)
        #mlflow.log_metric("rmse", rmse)
        #mlflow.log_metric("r2", r2)
        #mlflow.log_metric("mae", mae)

        #mlflow.sklearn.log_model(lr, "model")
