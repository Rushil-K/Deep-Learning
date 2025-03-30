# **RNN-Based Sentiment Analysis and Movie Review Scraper**

## **Overview**

This project is a state-of-the-art sentiment analysis solution built using a Recurrent Neural Network (RNN). It incorporates a custom-built web scraper that dynamically extracts movie reviews from Metacritic. The collected data undergoes a robust preprocessing pipeline before being used to train an RNN model that classifies reviews as either positive or negative. This solution is designed for scalability and can be extended to other domains requiring sentiment-based classification.

## **Key Features**

* **Dynamic Web Scraper**: Efficiently scrapes Metacritic movie reviews, handling pagination and saving structured data in CSV format.  
* **Deep Learning Model**: Utilizes an RNN-based architecture to achieve high-accuracy sentiment classification.  
* **Text Preprocessing Pipeline**: Includes tokenization, text vectorization, and data cleaning to enhance model performance.  
* **Scalability**: Can be adapted to process large-scale datasets and extended for other NLP applications.  
* **Recommendation System**: Uses sentiment scores to provide meaningful movie recommendations.  
* **Fully Open-Source**: No reliance on paid APIs, ensuring accessibility and transparency.

## **Setup Instructions**

### **Prerequisites**

Ensure you have the following dependencies installed before running the project:
```
pip install selenium beautifulsoup4 pandas numpy tensorflow keras scikit-learn
```
### **Running the Scraper**

1. Open `kkrk2127_movies_scraper_Metacritic.ipynb` in Jupyter Notebook.  
2. Run all cells to execute the scraper, which will fetch and save movie reviews to `metacritic_reviews.csv`.

### **Training and Using the Sentiment Analysis Model**

1. Load the `IMDB Dataset.csv` and `metacritic_reviews.csv` datasets.  
2. Preprocess the text data using the tokenizer (`tokenizer.pkl`).  
3. Train the RNN model and save it as `sentiment_analysis_model.h5`.  
4. Use the trained model to classify new reviews and analyze sentiment trends.

### **Extending the Project**

* **Enhance Model Performance**: Experiment with LSTM and GRU layers for improved accuracy.  
* **Deploy as a Web App**: Integrate with a frontend using Flask or Streamlit for real-time sentiment analysis.  
* **Expand Data Sources**: Extend the scraper to additional review platforms like IMDB or Rotten Tomatoes.

## **License**

This project is licensed under the [Apache 2.0 License](https://github.com/Rushil-K/Deep-Learning/blob/main/LICENSE).

## **Contributors**

* **Rushil Kohli** – [Follow Rushil](https://github.com/Rushil-K)  
* **Khushi Kalra** – [Follow Khushi](https://github.com/KhushiKalra21)

For further details, reach out via GitHub or LinkedIn.
