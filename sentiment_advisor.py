from datetime import date, timedelta
from pathlib import Path
import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk


class SentimentAdvisor:

    DATALAKE_PATH = Path('datalake')
    NEWS_DATA_FILE_PATH = DATALAKE_PATH / Path('news_dump.json')
    OUTPUT_PATH = Path('output')

    def __init__(self):
        self.data = pd.DataFrame()
        self.OUTPUT_PATH.mkdir(exist_ok=True)


    # extract data from excel file
    def ETL(self):
        print('#'*10, 'Extracting news data from excel file','#'*10)

        # read news from excel file
        if self.data.empty:
            self.data = pd.DataFrame()
        self.data = pd.read_json(self.NEWS_DATA_FILE_PATH)
        self.data.rename(columns = {'Date/Time': 'Date'}, inplace=True)
        self.data.set_index('Date', inplace=True)
        self.data.index = pd.to_datetime(self.data.index)
        self.data.sort_index(inplace = True, ascending=False)

        print('Extracted:  ', self.data.columns.tolist())
        print('Equity List:', self.data['Equity'].unique())


    # select relevant features required for model
    def feature_engineering(self):
        print()
        print('#'*10, 'Selecting Relevant Features', '#'*10)
        assert not self.data.empty, 'No data has been extracted'

        # remove non relevant features
        features = ['Equity', 'Headline']
        self.data = self.data[self.data.columns.intersection(features)]

        print('Selected features:', features)


    # run and test model using external headlines first
    def model_design(self):
        print()
        print('#'*10, 'Testing Sentiment Advisor using Separate News Source', '#'*10)

        neg_articles = DataFrame.from_dict({
            '2021-08-02 15:59:00': {
                'Equity': 'CLP',
                'Headline': 'Yallourn flooding hits EnergyAustralia profits'
            }, 
            '2021-08-06 09:55:00': {
                'Equity': 'ALB',
                'Headline': 'Lithium giant’s lament over labour shortages, cost blowouts in WA' 
            }, 
            '2021-07-30 07:24:00': {
                'Equity': 'AMZN',
                'Headline': 'Amazon forecast misses estimates as pandemic online bump fades' 
            }
        }, orient='index')
        pos_articles = DataFrame.from_dict({
            '2021-08-08 14:39:00': {
                'Equity': 'TLS',
                'Headline': 'Telstra secures deal to buy GP software group MedicalDirector' 
            }, 
            '2021-08-04 05:00:00': {
                'Equity': 'SYD',
                'Headline': 'Sydney Airport gets green light for new hotel precinct' 
            }, 
            '2021-08-04 13:44:00': {
                'Equity': 'GUD',
                'Headline': 'Profit boom as drivers hang onto cars longer'
            }
        }, orient='index')
        neu_articles = DataFrame.from_dict({
            '2021-08-04 09:18:00': {
                'Equity': 'WBC',
                'Headline': 'Westpac launches western Sydney jab program'
            }, 
            '2021-07-28 05:00:00': {
                'Equity': 'Nexport',
                'Headline': 'Nexport flags joint EV site with Tritium'
            }, 
            '2021-08-05 17:00:00': {
                'Equity': 'LG',
                'Headline': 'LG wants consumers to ‘rediscover good’ after rough year'
            }
        }, orient='index')

        neg_articles.index.rename('Date', inplace=True)
        pos_articles.index.rename('Date', inplace=True)
        neu_articles.index.rename('Date', inplace=True)
        data = pd.concat([neg_articles, pos_articles, neu_articles])
        data.index = pd.to_datetime(data.index)

        self.calculate_sentiment(
            data = data, 
            equity_list = data['Equity'].unique(), 
            output_suffix = '_simulated'
        )


    # run model using our actual data
    def model_implementation(self):
        print()
        print('#'*10, 'Running Sentiment Advisor using Extracted News Source', '#'*10)
        self.calculate_sentiment(
            data = self.data, 
            equity_list = self.data['Equity'].unique()
        )


    # use vader lexicon to calculate sentiment of news articles
    def calculate_sentiment(self, data, equity_list, output_suffix: str = ''):
        assert not data.empty, 'No data has been provided'
        assert 'Headline' in data, 'No abstracts in provided data'

        # calculate compound sentiment per headline
        vader = SentimentIntensityAnalyzer()
        headline_sentiment = data['Headline'].apply(vader.polarity_scores).map(
            lambda sentiment: sentiment['compound']
        )
        headline_sentiment = data.join(
            pd.DataFrame(headline_sentiment.values, index=headline_sentiment.index, columns=['Sentiment'])
        )
        headline_sentiment.sort_index(inplace = True, ascending=False)
        with pd.ExcelWriter(self.OUTPUT_PATH / Path('headline_sentiment'+output_suffix+'.xlsx')) as writer:
            headline_sentiment.to_excel(writer)

        # categorise sentiment per headline as negative, neutral, or positive
        headline_sentiment_categorised = headline_sentiment.copy()
        headline_sentiment_categorised['Sentiment'] = pd.cut(
            headline_sentiment_categorised['Sentiment'], 
            bins = [-1, -0.3, 0.3, 1], 
            labels = ['Negative', ' ', 'Positive']
        )
        with pd.ExcelWriter(self.OUTPUT_PATH / Path('headline_sentiment_categorised'+output_suffix+'.xlsx')) as writer:
            headline_sentiment_categorised.to_excel(writer)

        # calculate sentiment per equity
        with pd.ExcelWriter(self.OUTPUT_PATH / Path('equity_sentiment'+output_suffix+'.xlsx')) as writer:
            equity_sentiment = {}
            for equity in equity_list:
                equity_sentiment[equity] = headline_sentiment.loc[headline_sentiment['Equity'] == equity]
                equity_sentiment[equity].to_excel(writer, sheet_name=equity)
        
        # categorise sentiment per equity as negative, neutral, or positive
        with pd.ExcelWriter(self.OUTPUT_PATH / Path('equity_sentiment_categorised'+output_suffix+'.xlsx')) as writer:
            equity_sentiment_categorised = {}
            for equity in equity_list:
                equity_sentiment_categorised[equity] = equity_sentiment[equity].copy()
                equity_sentiment_categorised[equity]['Sentiment'] = pd.cut(
                    equity_sentiment_categorised[equity]['Sentiment'], 
                    bins = [-1, -0.3, 0.3, 1], 
                    labels = ['Negative', ' ', 'Positive']
                )
                equity_sentiment_categorised[equity].to_excel(writer, sheet_name=equity)

        # plot daily sentiment
        sentiment_daily = [
            round(
                sentiment['Sentiment'].loc[
                    sentiment['Sentiment'].index > sentiment['Sentiment'].index[0] - timedelta(days=1)
                ].sum(), 3
            ) for sentiment in equity_sentiment.values()
        ]
        limit = max(abs(max(sentiment_daily))+0.1, abs(min(sentiment_daily))+0.1)
        plt.ylim(-limit, limit)
        plt.bar(equity_sentiment.keys(), sentiment_daily)
        plt.savefig(self.OUTPUT_PATH / Path('sentiment_daily'+output_suffix))
        plt.close()

        # plot weekly sentiment
        sentiment_weekly = [
            round(
                sentiment['Sentiment'].loc[
                    sentiment['Sentiment'].index > sentiment['Sentiment'].index[0] - timedelta(days=7)
                ].sum(), 3
            ) for sentiment in equity_sentiment.values()
        ]
        limit = max(abs(max(sentiment_weekly))+0.1, abs(min(sentiment_weekly))+0.1)
        plt.ylim(-limit, limit)
        plt.bar(equity_sentiment.keys(), sentiment_weekly)
        plt.savefig(self.OUTPUT_PATH / Path('sentiment_weekly'+output_suffix))
        plt.close()

        print(headline_sentiment_categorised)
        print('-'*20)
        print('Recent daily sentiment: ', dict(zip(equity_list, sentiment_daily)))
        print('Recent weekly sentiment:', dict(zip(equity_list, sentiment_weekly)))