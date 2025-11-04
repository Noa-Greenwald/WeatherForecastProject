import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def load_and_clean_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df.dropna(inplace=True)
    df['Month'] = df['Date'].dt.month
    return df

def calculate_monthly_averages(df):
    temp_avg = df.groupby('Month')['Temperature'].mean().to_numpy()
    humidity_avg = df.groupby('Month')['Humidity'].mean().to_numpy()
    return temp_avg, humidity_avg

def plot_monthly_temps(monthly_avg_temp):
    months = range(1, 13)
    plt.plot(months, monthly_avg_temp, marker='o')
    plt.title('ממוצע טמפרטורות חודשי')
    plt.xlabel('חודש')
    plt.ylabel('טמפרטורה')
    plt.grid(True)
    plt.show()

def plot_season_boxplot(df):
    sns.boxplot(x='Season', y='Temperature', data=df)
    plt.title('Boxplot לפי עונות')
    plt.show()

def train_and_evaluate_model(df):
    X = df[['Temperature', 'Humidity']]
    y = df['Season']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"דיוק המודל: {accuracy:.2f}")
