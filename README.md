

## CPSC393 Final Project

A comprehensive stock market prediction system utilizing an ensemble of machine learning techniques, including LSTM, GRU, Transformer, and Prophet models. Designed to analyze and predict stock prices, this project showcases a comparative study of different algorithms and their effectiveness in predicting financial market trends, while leveraging the power of ensemble methods for improved accuracy and robustness.

## Prerequisites

Python 3.8+
Pandas
NumPy
Scikit-learn
Scikit-optimize
TensorFlow
Keras
Statsmodels
Matplotlib
Docker

### Setup and Usage
This section combines the instructions for installing Docker and using the Jupyter notebook 'Testing.ipynb' to train and evaluate the models.

### Docker Installation
Install Docker on your machine if you haven't already.
[Docker Desktop Download](https://www.docker.com/products/docker-desktop/)
### Building the Docker Image
To build the Docker image, navigate to the project directory and run the following command:

```bash
docker build -t cpsc393_stock_pred .
```
### Running the Docker Container
After building the Docker image, you can run the project using the following command:

```bash
docker run -it --rm -p 8000:8000 cpsc393_stock_pred
```
runs the main.py file

### Shutting Down the Docker Container
To shut down the Docker container, press Ctrl+C in the terminal.


- **Always remember to shut down the Docker container** when you are done using it. If you do not shut down the container, it will continue to run in the background and use up your computer's resources.

### file structure

├── main.py
├── Models/
│   ├── __init__.py
│   ├── lstm_model.py
│   ├── gru_model.py
│   ├── arima_model.py
│   └── transformer_model.py
├── Data/
│   ├── __init__.py
│   ├── data_collector.py
│   └── data_handler.py
└── Predictors/
    ├── __init__.py
    ├── base_predictor.py
    ├── lstm_predictor.py
    ├── gru_predictor.py
    ├── arima_predictor.py
    └── transformer_predictor.py


### Legal

Disclaimer: Student Project, Not Intended for Actual Use

This software is a student project developed for educational purposes only. It is not intended to be used for any commercial or professional purpose, including but not limited to trading, investment, or financial decision-making.

The creators of this software, including AM, HF, JM, and TG, as well as their affiliated educational institution, make no representations or warranties, express or implied, regarding the accuracy, reliability, or completeness of the information or predictions provided by this software.

By using this software, you acknowledge that the creators and their affiliated educational institution shall not be held liable for any errors, omissions, or inaccuracies in the information provided, nor any direct, indirect, incidental, consequential, special, or exemplary damages resulting from your reliance on or use of this software.

You also acknowledge that this software should not be used as a substitute for professional financial advice, and any investment or financial decisions should be made after consulting with a qualified financial professional.