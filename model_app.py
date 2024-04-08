import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from pickle import dump, load


def open_data(path="data/data_loan"):
    df = pd.read_csv('data/data_loan.csv')
    df = df[['Gender', 'Married', 'Dependents',
        'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome',
        'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area',
        'Loan_Status']]
    return df


def split_data(df):
    y_df = df['Loan_Status']
    X_df = df.drop(columns=['Loan_Status'])
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.1, random_state=42)
    return X_train, X_test, y_train, y_test


def preprocess_data(df):
    df = df.dropna()
    binary_cols = ['Married', 'Loan_Status']
    categ_cols = ['Gender', 'Dependents', 'Education', 'Self_Employed', 'Loan_Amount_Term', 'Credit_History', 'Property_Area']

    for col in categ_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    for col in binary_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    X_train, X_test, y_train, y_test = split_data(df)
    return X_train, X_test, y_train, y_test


def fit_and_save_model(X_train, X_test, y_train, y_test, path='data/model_weights.mw'):
    model = RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_leaf=10, random_state=42)
    model.fit(X_train, y_train)
    test_prediction = model.predict(X_test)
    accuracy = accuracy_score(test_prediction, y_test)
    print(f"Model accuracy is {accuracy:.2f}")
    with open(path, "wb") as file:
        dump(model, file)
    print(f"Model was saved to {path}")

def load_model_and_predict(df, path='data/model_weights.mw'):
    with open(path, "rb") as file:
        model = load(file)
    prediction = model.predict(df)[0]
    prediction_proba = model.predict_proba(df)[0]
    encode_prediction_proba = {
        0: "Вам не повезло",
        1: "Вам повезло"
    }
    encode_prediction = {
        0: "Увы, банк отказал Вам в кредите",
        1: "Ура! Вам одобрен кредит!"
    }
    prediction_data = {}
    for key, value in encode_prediction_proba.items():
        prediction_data.update({value: prediction_proba[key]})
    prediction_df = pd.DataFrame(prediction_data, index=[0])
    prediction = encode_prediction[prediction]
    return prediction, prediction_df


if __name__ == "__main__":
    df = open_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    fit_and_save_model(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    df = open_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    fit_and_save_model(X_train, X_test, y_train, y_test)
