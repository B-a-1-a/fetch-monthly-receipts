import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# Linear Regression 
def linear_regression(X, y):
    # Add bias term
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    # Calculate the optimal theta using the normal equation
    theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    return theta_best

def linear_forecast(theta, X_future):
    # Bias term to future data 
    X_future_b = np.c_[np.ones((X_future.shape[0], 1)), X_future]
    # Return the predicted values
    return X_future_b @ theta

def load_and_prepare_data():
    df = pd.read_csv('data_daily.csv') 
    # column with a daily frequency starting from 2021 Jan 1
    df['Date'] = pd.date_range(start='2021-01-01', periods=len(df), freq='D')
    
    # Date column as the index of the DataFrame
    df.set_index('Date', inplace=True)
    
    if 'Receipt_Count' not in df.columns:
        raise ValueError("'Receipt_Count' column not found in the CSV file.")
    
    return df

def train_linear_model(data):
    # Time is used as the feature, and receipt count as the target
    X, y = np.arange(len(data)).reshape(-1, 1), data
    # Train the linear regression model
    theta = linear_regression(X, y)
    return theta

def calculate_monthly_predictions(linear_forecasts):
    # Initialize a list to store monthly predictions
    monthly_predictions = []
    # For each month (1 to 12), calculate the total predicted receipts
    for month in range(1, 13):
        # Approximate each month as 30 days for simplicity
        start_day = (month - 1) * 30
        end_day = month * 30
        # Sum the daily forecasts for each month
        monthly_pred = np.sum(linear_forecasts[start_day:end_day])
        # Append the total for the month to the predictions list
        monthly_predictions.append(int(monthly_pred))
    return monthly_predictions

def train_and_forecast():
    # Load and prepare the receipt data
    df = load_and_prepare_data()
    # Extract the 'Receipt_Count' column as the target variable
    data = df['Receipt_Count'].values
    
    # Train the linear regression model on the existing data
    linear_theta = train_linear_model(data)
    # Prepare future time steps for forecasting (365 days into the future)
    future_X = np.arange(len(data), len(data) + 365).reshape(-1, 1)
    # Generate predictions for the future using the trained model
    linear_forecasts = linear_forecast(linear_theta, future_X)

    # Calculate monthly predictions based on the daily forecasts
    monthly_predictions = calculate_monthly_predictions(linear_forecasts)
    return monthly_predictions, df, linear_forecasts

# Streamlit UI - Title for the app
st.title("Receipt Scanner Prediction for 2022")

# Train the model
monthly_predictions, df, linear_forecasts = train_and_forecast()

month_names = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
}

# Dropdown 
selected_month = st.selectbox('Select a month:', list(month_names.keys()))

# Numeric index for the selected month
selected_month_index = month_names[selected_month]

# Predicted total number of receipts for the selected month
st.write(f'Predicted total number of scanned receipts for {selected_month}: {monthly_predictions[selected_month_index - 1]}')

# DataFrame for the forecasted data (dates and predicted receipts)
forecast_dates = pd.date_range(start=df.index[-1], periods=len(linear_forecasts)+1, freq='D')[1:]
forecast_df = pd.DataFrame({'Date': forecast_dates, 'Predicted Receipts': linear_forecasts})

# Altair selections for brushing and multi-click interactions
brush = alt.selection_interval(encodings=['x']) 
click = alt.selection_multi(encodings=['color'])  # Multi-click 

# Define a color scale for months
color_scale = alt.Scale(domain=list(month_names.keys()), range=["#e7ba52", "#a7a7a7", "#aec7e8", "#1f77b4", "#9467bd", "#ff7f0e", "#98df8a", "#d62728", "#ff9896", "#c5b0d5", "#c49c94", "#f7b6d2"])

# Create a scatter plot for the actual data points
actual_scatter = alt.Chart(df.reset_index()).mark_point(size=50).encode(
    x='Date:T',  # Date on the x-axis
    y=alt.Y('Receipt_Count:Q', scale=alt.Scale(domain=[5_000_000, 15_000_000])),  # Receipt count on the y-axis
    color=alt.condition(brush, alt.Color('month(Date):N', scale=color_scale), alt.value('lightgray')),  # Apply color scale to months
    tooltip=['Date:T', 'Receipt_Count:Q']  # Tooltips for interaction
)

# Create a line chart for the predicted receipts
prediction_line = alt.Chart(forecast_df).mark_line(strokeWidth=3, color='orange').encode(
    x='Date:T',  # Date on the x-axis
    y=alt.Y('Predicted Receipts:Q'),  # Predicted receipt count on the y-axis
    tooltip=['Date:T', 'Predicted Receipts:Q']  # Tooltips for interaction
)

# Layer the actual scatter plot and prediction line chart together
combined_chart = alt.layer(actual_scatter, prediction_line).properties(
    width=700,
    height=400,
    title="Actual And Predicted Receipt Data"
).add_selection(brush).transform_filter(click)  

# Bar chart showing the average receipt count for the brushed months
average_monthly_receipts = (
    alt.Chart(df.reset_index())
    .mark_bar()
    .encode(
        x=alt.X('month(Date):N', title='Month'),  # Month on the x-axis
        y=alt.Y('mean(Receipt_Count):Q', title='Average Receipt Count'),  # Average receipt count on the y-axis
        color=alt.Color('month(Date):N', scale=color_scale),  # Apply color scale
        tooltip=['month(Date):N', 'mean(Receipt_Count):Q']  # Tooltips for interaction
    )
    .properties(width=700, height=200, title='Average Monthly Receipt Count for Selected Range')
    .transform_filter(brush)  # Apply the brush filter
)

# Combine the charts with the bar chart
combined_chart_with_bars = alt.vconcat(combined_chart, average_monthly_receipts)

# Streamlit Switching between themes
tab1, tab2 = st.tabs(["Streamlit theme (default)", "Altair native theme"])

with tab1:
    # Streamlit theme
    st.altair_chart(combined_chart_with_bars, theme="streamlit", use_container_width=True)
with tab2:
    # native Altair
    st.altair_chart(combined_chart_with_bars, theme=None, use_container_width=True)
