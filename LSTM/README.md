# Stock Price Analysis and Prediction

[REPORT](https://colab.research.google.com/drive/147txJ6qZSg4vlEZtzna-0t5dO7D9lrB0?usp=sharing) 

## Project Description

This project provides a comprehensive analysis of stock price dynamics, with a focus on the NIFTY 50 Index and ICICI Bank. The project aims to develop and evaluate machine learning models for predicting stock prices, and to provide insights into market behavior.  The analysis considers the complexities of stock market data, including volatility, cyclical patterns, and the influence of various economic and market factors.  The ultimate goal is to equip investors with tools and knowledge to make more informed decisions.

## Data

### NIFTY 50 Data
The data describes the **NIFTY 50 index**, a key benchmark for the Indian stock market. It represents the weighted average of the 50 largest Indian companies listed on the National Stock Exchange (NSE), providing a snapshot of the market's performance. The NIFTY 50 is widely used by investors, analysts, and portfolio managers to track the overall direction of the Indian equity market. It serves as the basis for various investment instruments, such as index funds and derivatives. Each row in the dataset represents a specific point in time.

For more information about the NIFTY 50, you can visit the official NSE website: [web](https://www.nseindia.com/market-data/live-equity-market)

### ICICI Bank Data
* The project also includes data for ICICI Bank.  Detailed information about ICICI Bank can be found on the official website: [web](https://www.icicibank.com/)
* The ICICI Bank data has same columns as NIFTY 50 data.

## Methodology

The project employs the following methodology:

1.  **Data Acquisition and Preprocessing**: Historical stock price data for the NIFTY 50 Index and ICICI Bank is acquired from the Eikon database. The data is then preprocessed to ensure data quality.  This involves cleaning the data, handling missing values, and formatting it for analysis.
2.  **Time-Series Decomposition**:  Time-series decomposition is used to separate the data into trend, seasonal, and residual components.
3.  **Model Development**:  Machine learning models, including LSTM and Random Forest, are developed to predict future stock prices.
4.  **Model Training and Validation**: The models are trained on historical data and validated using appropriate techniques to prevent overfitting and ensure generalization.
5.  **Performance Evaluation**:  The performance of the models is evaluated using relevant metrics.
6.  **Forecast Generation**:  The trained models are used to generate short-term price forecasts.
7.  **Recommendation Generation**:  Investment recommendations (e.g., Buy, Sell, Hold) are generated based on the model predictions.

## Key Findings

* The analysis reveals significant volatility and cyclical patterns in stock prices.
* The LSTM model demonstrates a reasonable ability to capture the general trend and a portion of the volatility.
* A Random Forest model outperforms the LSTM model in this specific application.
* Intraday and weekly patterns are identified.
* Short-term forecasts for the NIFTY 50 Index and ICICI Bank are provided.

## Investment Implications

The findings have implications for investors, including:

* Informed decision-making through the use of predictive models.
* Improved risk management by understanding stock price volatility.
* Potential profit maximization by identifying buying and selling opportunities.

## Future Scope and Improvements

The project can be extended in several ways:

* Enhanced feature engineering with additional technical and fundamental indicators.
* Exploration of hybrid models, such as combining CNNs and LSTMs.
* Development of a recommendation system to provide personalized investment advice.
* Integration of real-time data and adaptive modeling.
* Incorporation of more sophisticated risk management techniques.
* Implementation of Explainable AI (XAI) to improve the transparency of the models.

## Contributors

* Rushil Kohli - [Follow](https://github.com/Rushil-K)
* Khushi Kalra - [Follow](https://github.com/KhushiKalra21)

## License

This project is licensed under the [Apache License 2.0](https://github.com/Rushil-K/Deep-Learning/blob/main/LICENSE).

