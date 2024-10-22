from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests

def get_sentiment_scores(news_headlines):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = [analyzer.polarity_scores(headline)['compound'] for headline in news_headlines]
    return sentiment_scores

def fetch_news_sentiment():
    
    news_api_url = 'https://newsapi.org/v2/everything?q=Apple&apiKey=YOUR_API_KEY'
    response = requests.get(news_api_url)
    headlines = [article['title'] for article in response.json()['articles']]
    sentiment_scores = get_sentiment_scores(headlines)
    return sentiment_scores

def preprocess_data(file_path):
    
    data = pd.read_csv(file_path)
    
    
    sentiment_scores = fetch_news_sentiment()
    data['Sentiment'] = sentiment_scores
    
   
    data['RSI'] = ta.rsi(data['Close'], length=14)
    data['Bollinger_Bands'] = ta.bbands(data['Close'])

   
    data.to_csv('data/processed_with_sentiment.csv')
    return data
