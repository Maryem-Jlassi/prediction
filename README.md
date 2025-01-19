# Streamlit App for ML Model Deployment
This repository contains a Streamlit app designed to deploy and interact with machine learning models. The app allows users to interact with various predictive models related to food safety, employee compliance, and restaurant risk assessment within the restaurant industry in Los Angeles.
## **Features**
Predictive Models:

 -Food Quality Score Prediction

 -Employee Compliance Rate Prediction

 -Restaurant Risk Level Prediction

Business Intelligence: Display of integrated Power BI dashboards.

Data Insights: Detailed analytics on restaurant inspections and food safety, including map visualizations.

## **Technologies Used**
Streamlit: Framework to build the interactive web app.

Machine Learning Libraries:

scikit-learn for data preprocessing, feature selection, and evaluation.

XGBoost for predictive models.


joblib for saving/loading models.

Folium: For displaying interactive maps with geographic data.

Power BI: Integrated for business analytics dashboards.
## **Installation**
**Prerequisites**
Before running the app, ensure you have the following libraries installed:

Python 3.x

Streamlit

scikit-learn

XGBoost

pandas

numpy

matplotlib

joblib

folium

To install the required libraries, you can use the following command:

pip install streamlit scikit-learn xgboost pandas numpy matplotlib joblib folium

**Clone the Repository**
git clone https://github.com/Maryem-Jlassi/prediction.git

cd prediction

**Running the App**

To run the app, navigate to the directory where the repository is located and execute the following command:

streamlit run project.py

This will launch the app in your default web browser.


## **App Pages**

-Main Page: Provides an overview of the platform, featuring links to predictive analytics, business intelligence, and data insights.

-Predictive Models Page: Choose between three predictive models related to food quality, employee compliance, and restaurant risk level.

-Power BI Page: Embedded Power BI analytics dashboard for deeper insights into the restaurant inspection data.

-Data Insights Page: Displays key data insights and a map showing restaurant inspection locations.

## **Contributing**

Feel free to fork this project, open issues, and submit pull requests. Contributions are always welcome!

## **License**

This project is licensed under the MIT License - see the LICENSE file for details.

