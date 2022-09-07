import numpy
import pandas as pd
import openpyxl
import xlsxwriter
import nltk
from datetime import datetime
from statistics import *
import numpy as np
from textblob import TextBlob

bitcoinPrices = pd.DataFrame()
smallTweets = pd.DataFrame()
newsHeadlines = pd.DataFrame()
# Read in Bitcoin historical price CSV (collected from Bitstamp exchange),
# skip first row and make second row column headers
data = pd.read_csv("Datasets/Bitstamp_BTCUSD_d.csv", skiprows=0, header=1)
newsHeadlinesData = pd.read_csv("Datasets/Headline_Crypto.csv")
# tweets = pd.read_csv("Datasets/tweets.csv", delimiter=';', skiprows=0, lineterminator='\n' )
smallTweetData = pd.read_csv('Datasets/tweets.csv', delimiter=';', skiprows=0, lineterminator='\n')

bitcoinPrices = pd.DataFrame(data)
# tweetData = pd.DataFrame(tweets)
smallTweets = pd.DataFrame(smallTweetData[['timestamp', 'text\r']])
smallTweets
newsHeadlines = pd.DataFrame(newsHeadlinesData)

smallTweets[:5]

tweets = smallTweets['text\r']


def calc_sentiment(txt):
    blob = TextBlob(txt)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

tweets_sentiments = tweets.apply(calc_sentiment)               # calc sentiment polarity & subjectivity, return in a Series of tuples
tweets_polarity = tweets_sentiments.apply(lambda x: x[0])      # new column of polarity
tweets_subjectivity = tweets_sentiments.apply(lambda x: x[1])  # new column of subjectivity

smallTweets['polarity'], smallTweets['subjectivity'] = tweets_polarity, tweets_subjectivity  # create the series

smallTweets.sample(10)        # display 10 random rows

# Write tweets dataframe with polarity and subjectivity to new csv
smallTweets.to_csv(index=False)

print(bitcoinPrices[:5])
# print (tweetData[:5])
print(smallTweetData[:5])
print(newsHeadlines[:5])


# print(smallTweets.iloc[['2019-11-23']])

# Convert Date columns for bitcoin, tweets, and news headlines to same date_time format
# This will allow for merging the dataframes by matching dates in each row
newsHeadlines['Date'] = pd.to_datetime(newsHeadlines['Date'])
newsHeadlines['Date'] = newsHeadlines['Date'].apply(lambda t: t.strftime('%Y-%m-%d'))

smallTweets['timestamp'] = pd.to_datetime(smallTweets['timestamp'])
smallTweets['timestamp'] = smallTweets['timestamp'].apply(lambda t: t.strftime('%Y-%m-%d'))
smallTweets = smallTweets.sort_values('timestamp')
# Reset index after sorting
smallTweets.reset_index(drop=True, inplace=True)
smallTweets[:-5]

bitcoinPrices['date'] = pd.to_datetime(bitcoinPrices['date'])
bitcoinPrices['date'] = bitcoinPrices['date'].apply(lambda t: t.strftime('%Y-%m-%d'))


# Determine minimum and maximum possible dates for analysis
print('Min Date for Headlines', min(newsHeadlines['Date']))
print('Max Date for Headlines', max(newsHeadlines['Date']))
print('Min Date for Tweets', min(smallTweets['timestamp']))
print('Max Date for Tweets', max(smallTweets['timestamp']))
print('Min Date for Bitcoin', min(bitcoinPrices['date']))
print('Max Date for Bitcoin', max(bitcoinPrices['date']))

# Feasible Date Range -> 2015-2018 min: 2014-11-28 max: 2018-05-04

# Calculate log difference between opening and closing btc price for each day
bitcoinPrices['log_diff'] = np.log(bitcoinPrices['close']) - np.log(bitcoinPrices['open'])
bitcoinPrices['log_diff']

# Group by date
smallTweets['timestamp'][:-5]
smallTweets = smallTweets.sort_values('timestamp')
groupedTweets = smallTweets
groupedTweets = groupedTweets.groupby(['timestamp'])['text\r'].apply(','.join).reset_index()
groupedTweets[:-5]

# Categorical value for whether price movement on the day was positive or negative, goal will be to train model
# to predict which direction price might go
bitcoinPrices['movement'] = [1 if log_diff > 0 else 0 for log_diff in bitcoinPrices['log_diff']]

# For each row in tweet csv, tokenize text, then push tokenized text to list and outer join with
# matching timestamp in bitcoinData dataframe
# for index, tweet in smallTweets.iterrows():
#     tweetText = tweet['text\r']
#     tweetTime = tweet['timestamp']
#     # print(tweet['timestamp'], tweet['text\r'])
#     # Convert string of date created from tweet to datetime
#     dateCompare = tweetTime
#     new_datetime = datetime.strptime(
#         datetime.strftime(datetime.strptime(dtime, '%a %b %d %H:%M:%S +0000 %Y'), '%Y-%m-%d'), "%Y-%m-%d")
#     if new_datetime > dateCompare:
#         fourthTweets.append(tweet)
#     else:
#         thirdTweets.append(tweet)
#     print(tweetTime)

# for tweet in smallTweets:
#     tweetText = tweet
#     print(tweetText)