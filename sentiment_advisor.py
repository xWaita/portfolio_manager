from datetime import date
from pathlib import Path
import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np
import matplotlib.pyplot as plt

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')


class SentimentAdvisor:

    DATALAKE_PATH = Path('datalake')
    NEWS_DATA_FILE_PATH = DATALAKE_PATH / Path('news_dump.json')
    OUTPUT_PATH = Path('output')

    def __init__(self):
        self.data = pd.DataFrame()
        self.equity_list = set()
        self.OUTPUT_PATH.mkdir(exist_ok=True)

    def ETL(self):
        print('#'*10, 'Extracting data from excel', self.NEWS_DATA_FILE_PATH,'#'*10)

        # read news from excel file
        if self.data.empty:
            self.data = pd.DataFrame()
        self.data = pd.read_json(self.NEWS_DATA_FILE_PATH)
        self.data.rename(columns = {'Date/Time': 'Date'}, inplace=True)
        self.data.set_index('Date', inplace=True)
        self.data.index = pd.to_datetime(self.data.index)
        self.data.sort_index(inplace = True)

    def feature_engineering(self):
        print()
        print('#'*10, 'Selecting relevant features', '#'*10)
        assert not self.data.empty, 'No data has been extracted'

        # remove non relevant features
        features = ['Equity', 'Headline']
        self.data = self.data[self.data.columns.intersection(features)]

        # determine which equities have news
        if self.equity_list:
            self.equity_list.clear()
        self.data['Equity'].map(self.equity_list.add)

    def model_design(self):
        self._calculate_sentiment(self)
    
    def model_implementation(self):
        self._calculate_sentiment(self.data)
        
    def _calculate_sentiment(self, data):
        print()
        print('#'*10, 'Calculating Sentiment', '#'*10)
        assert not data.empty, 'No data has been provided'
        assert 'Headline' in data, 'No abstracts in provided data'

        # calculate compound sentiment per article
        vader = SentimentIntensityAnalyzer()
        headline_sentiment = data['Headline'].apply(vader.polarity_scores).map(
            lambda sentiment: sentiment['compound']
        )
        headline_sentiment = data.join(
            pd.DataFrame(headline_sentiment.values, index=headline_sentiment.index, columns=['Sentiment'])
        )
        with pd.ExcelWriter(self.OUTPUT_PATH / Path('headline_sentiment.xlsx')) as writer:
            headline_sentiment.to_excel(writer)

        # categorise compound sentiment as negative, neutral or positive
        simplified_sentiment = headline_sentiment.copy()
        simplified_sentiment['Sentiment'] = pd.cut(
            simplified_sentiment['Sentiment'], 
            bins = [-1, -0.3, 0.3, 1], 
            labels = ['Negative', 'Neutral', 'Positive']
        )
        with pd.ExcelWriter(self.OUTPUT_PATH / Path('simplified_sentiment.xlsx')) as writer:
            simplified_sentiment.to_excel(writer)

        # calculate sentiment per date
        date_sentiment = headline_sentiment
        date_sentiment.index = date_sentiment.index.strftime('%d/%m/%Y')
        date_sentiment = date_sentiment.groupby(date_sentiment.index).mean()
        with pd.ExcelWriter(self.OUTPUT_PATH / Path('date_sentiment.xlsx')) as writer:
            date_sentiment.to_excel(writer)

        # calculate sentiment per equity
        with pd.ExcelWriter(self.OUTPUT_PATH / Path('equity_sentiment.xlsx')) as writer:
            equity_sentiment = {}
            for equity in self.equity_list:
                equity_sentiment[equity] = headline_sentiment \
                    .loc[headline_sentiment['Equity'] == equity] \
                    .drop('Equity', axis=1)
                equity_sentiment[equity].to_excel(writer, sheet_name=equity)

        # plot daily sentiment
        daily_sentiment = [sentiment['Sentiment'][0] for sentiment in equity_sentiment.values()]
        limit = max(abs(max(daily_sentiment))+0.1, abs(min(daily_sentiment))+0.1)
        plt.ylim(-limit, limit)
        plt.bar(equity_sentiment.keys(), daily_sentiment)
        plt.savefig(self.OUTPUT_PATH / Path('daily_sentiment'))
        plt.show()
        plt.close()

        # plot weekly sentiment
        weekly_sentiment = [sentiment['Sentiment'].iloc[0:7].mean() for sentiment in equity_sentiment.values()]
        print(weekly_sentiment)
        limit = max(abs(max(weekly_sentiment))+0.1, abs(min(weekly_sentiment))+0.1)
        plt.ylim(-limit, limit)
        plt.bar(equity_sentiment.keys(), weekly_sentiment)
        plt.savefig(self.OUTPUT_PATH / Path('weekly_sentiment'))
        plt.show()
        plt.close()