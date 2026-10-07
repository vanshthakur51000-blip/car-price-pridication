# 🚗 Car Price Prediction

## 📌 About the Project

This project predicts the selling price of used cars using Machine Learning.

The project uses car details such as present price, manufacturing year, driven kilometers, fuel type, selling type, transmission, and previous owners.

A Streamlit web application is created to make predictions easily.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit

## 🤖 Machine Learning Models

Two models were tested:

- Linear Regression — R² Score: 0.767
- Random Forest — R² Score: 0.432

**Best Model: Linear Regression**

## 📊 Dataset

The dataset contains information about used cars.

A new feature called `Car_Age` was created from the `Year` column.

## 🌐 Web Application

The Streamlit application allows users to enter car details and get an estimated selling price.

Example prediction:

**₹7.14 Lakhs**

## 📁 Project Structure

```text
car price pedication/
├── plots/
├── venv/
├── app.py
├── streamlit_app.py
├── car data.csv
├── car_price_model.pkl
├── README.md
└── requirements.txt