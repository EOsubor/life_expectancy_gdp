import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv('all_data.csv')

# Function to format large numbers with commas
def format_number(num):
    return "{:,.2f}".format(num)

# Streamlit app
st.title('Life Expectancy and GDP Analysis')
st.write('This app allows users to explore the relationship between GDP and life expectancy, and predict life expectancy based on GDP input.')

# Display dataset
st.subheader('Dataset')
st.write(data.head())

# Dataset statistics
st.subheader('Dataset Statistics')
st.write(data.describe())

# Scatter plot of GDP vs Life Expectancy
st.subheader('GDP vs Life Expectancy at Birth')
fig1, ax1 = plt.subplots()
sns.scatterplot(data=data, x='GDP', y='Life expectancy at birth (years)', hue='Country', ax=ax1)
st.pyplot(fig1)

# Prepare data for model
features = data[['GDP']]
target = data['Life expectancy at birth (years)']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Display performance metrics
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
st.subheader('Model Performance')
st.write(f'Mean Squared Error: {mse}')
st.write(f'R-squared: {r2}')

# Plot actual vs predicted
st.subheader('Actual vs Predicted Life Expectancy')
fig2, ax2 = plt.subplots()
sns.scatterplot(x=X_test['GDP'], y=y_test, label='Actual', color='blue', ax=ax2)
sns.lineplot(x=X_test['GDP'], y=predictions, label='Predicted', color='red', ax=ax2)
st.pyplot(fig2)

# Life expectancy prediction
st.subheader('Predict Life Expectancy')
input_gdp = st.number_input('Enter GDP (in USD):', min_value=float(data['GDP'].min()), max_value=float(data['GDP'].max()), value=float(data['GDP'].mean()))
predicted_life_expectancy = model.predict(np.array([[input_gdp]]))[0]
st.write(f'Predicted Life Expectancy: {predicted_life_expectancy:.2f} years')
st.write(f'Entered GDP: {format_number(input_gdp)} USD')
