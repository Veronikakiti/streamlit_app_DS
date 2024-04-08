import pandas as pd
import streamlit as st
from PIL import Image


st.set_page_config(
        layout="wide",
        initial_sidebar_state="auto",
        page_title="Кредитоспособность клиентов",
        page_icon="data/банк.jpg",

    )

st.header("Предсказание кредитоспособности клиента")
st.subheader("Определим, одобрит ли банк Вам кредит или нет")

def process_main_page():
    image = Image.open('data/банк.jpg')
    st.image(image)
    personal_info = process_side_bar_inputs()
    if st.button("Предсказать"):
        prediction, prediction_probas = load_model_and_predict(personal_info)
        write_prediction(prediction, prediction_probas)


def write_prediction(prediction, prediction_probas):
    st.write("## Предсказание")
    st.write(prediction)

    st.write("## Вероятность предсказания")
    st.write(prediction_probas)


def process_side_bar_inputs():
    st.sidebar.header('Укажите необходимые параметры')
    user_input_df = sidebar_input_features()
    st.write("## Ваши данные")
    st.write(user_input_df)

    user_input_df['ApplicantIncome'] = user_input_df['ApplicantIncome'] * 50
    user_input_df['CoapplicantIncome'] = user_input_df['CoapplicantIncome'] * 50
    user_input_df['LoanAmount'] = user_input_df['LoanAmount'] / 10
    user_input_df['Loan_Amount_Term'] = user_input_df['Loan_Amount_Term'] * 6
    return user_input_df


def sidebar_input_features():
    sex = st.sidebar.selectbox("Пол", ("Мужской", "Женский"))
    married = st.sidebar.selectbox("Замужем/женат", ("Нет", "Да"))
    education = st.sidebar.selectbox("Высшее образование", ("Есть", "Не имею"))
    dependents = st.sidebar.slider("Количество иждивенцев", min_value=0, max_value=10,
                                   value=0, step=1)
    self_employed = st.sidebar.selectbox("Вы самозанятый?", ("Нет", "Да"))
    income = st.sidebar.number_input("Ежемесячный доход заемщика, тыс. руб.", 0, 10000, 50)
    co_income = st.sidebar.number_input("Ежемесячный доход созаемщика, тыс. руб.", 0, 10000, 0)
    loan_amount = st.sidebar.number_input("Размер кредита, тыс. руб.", 0, 100000, 500)
    loan_term = st.sidebar.number_input("Срок кредита в месяцах", 0, 1200, 12)
    credit_history = st.sidebar.selectbox("Есть кредитная история", ("Нет", "Да"))
    area = st.sidebar.selectbox("Где проживает заемщик?", ("Город", "Поселок городского типа", "Сельская местность"))

    translatetion = {
        "Мужской": 1,
        "Женский": 0,
        "Есть": 1,
        "Да": 1,
        "Нет": 0,
        "Не имею": 0,
        "Город": 2,
        "Поселок городского типа": 1,
        "Сельская местность": 0
    }

    data = {
        "Gender": translatetion[sex],
        "Married": translatetion[married],
        "Dependents": dependents,
        "Education": abs(translatetion[education] - 1),
        "Self_Employed": translatetion[self_employed],
        "ApplicantIncome": income,
        "CoapplicantIncome": co_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": translatetion[credit_history],
        "Property_Area": translatetion[area]
    }

    df = pd.DataFrame(data, index=[0])
    return df


if __name__ == "__main__":
    process_main_page()
