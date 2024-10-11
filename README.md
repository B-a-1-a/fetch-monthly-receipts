
# Fetch Receipt Predictor

This application predicts the number of scanned receipts for each month in 2022 based on the daily receipt data from 2021. It features a linear regression model built from scratch and is containerized with Docker for easy deployment.

## Features

- **Machine Learning Model**: Built from scratch (without high-level libraries like scikit-learn).
- **Streamlit UI**: Allows user interaction to select months and view the predicted receipt counts.
- **Visualization**: Includes a chart of actual vs. predicted receipt counts and average monthly analysis.
- **Dockerized**: Easily build and run the app using Docker.

## Setup and Running Instructions

### Prerequisites

Ensure you have the following installed:

- [Docker](https://docs.docker.com/get-docker/)
- [Git](https://git-scm.com/)

### Steps

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/fetch-monthly-receipts.git
   cd fetch-receipt-predictor
   ```

2. **Build the Docker image:**

   ```bash
   docker build -t fetch-receipt-predictor .
   ```

3. **Run the Docker container:**

   ```bash
   docker run -p 8501:8501 fetch-receipt-predictor
   ```

4. **Access the app:**

   Open your browser and navigate to `http://localhost:8501`.

### Data

The dataset is provided in the `data` directory as `data_daily.csv`, and it includes daily receipt counts for 2021.

### Model Explanation

- **Model**: A simple linear regression model is trained on the daily receipt data from 2021.
- **Prediction**: The model forecasts daily receipt counts for 2022, and these predictions are aggregated to provide monthly totals.
- **Visualization**: The app visualizes actual receipt counts (2021) vs. predicted (2022) using interactive charts.

### Project Structure

```
fetch-receipt-predictor/
├── app.py                 # Main application file
├── Dockerfile             # Docker configuration
├── requirements.txt       # Python dependencies
├── data/
│   └── data_daily.csv     # Input dataset
└── README.md              # This file
```

### Dependencies

This project requires the following Python libraries:

- `streamlit`
- `pandas`
- `numpy`
- `altair`

To install them locally (without Docker), run:

```bash
pip install -r requirements.txt
```

### How It Works

- **Linear Regression**: The app implements a basic linear regression algorithm with the analytical approach with Normal Equation using NumPy. The algorithm fits a model to the 2021 receipt data to predict future values for 2022.
- **Visualization**: Interaction is provided through a simple web-app using Streamlit. The app also uses Altair to display interactive charts of actual vs. predicted receipt counts, with the option to zoom in on specific months.

### License
[MIT License](LICENSE).