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
dfTweets = pd.DataFrame()
dfHeadlines = pd.DataFrame()
# Read in Bitcoin historical price CSV (collected from Bitstamp exchange),
# skip first row and make second row column headers
btcData = pd.read_csv("Datasets/Bitstamp_BTCUSD_d.csv", skiprows=0, header=1)
headlineData = pd.read_csv("Datasets/Headline_Crypto.csv")
# tweets = pd.read_csv("Datasets/tweets.csv", delimiter=';', skiprows=0, lineterminator='\n' )
tweetData = pd.read_csv('Datasets/tweets.csv', delimiter=';', skiprows=0, lineterminator='\n')

bitcoinPrices = pd.DataFrame(btcData)
# tweetData = pd.DataFrame(tweets)
dfTweets = pd.DataFrame(tweetData[['timestamp', 'text\r']])
dfTweets
dfHeadlines = pd.DataFrame(headlineData)

# dfTweets[:5]
#
# tweets = dfTweets['text\r']

# print(bitcoinPrices[:5])
# print (tweetData[:5])
# print(tweetData[:5])
# print(dfHeadlines[:5])

# 10239 Drop rows from dfHeadlines dataframe after index 10,239 due to date formatting/missing value issues
print(dfHeadlines[:-5])
dfHeadlines.drop(dfHeadlines.index[10239:], inplace=True)
# Drop rows with missing date values
dfHeadlines.drop(dfHeadlines.index[2066:2070], inplace=True)
dfHeadlines.drop(dfHeadlines.index[7569:7589], inplace=True)
# print(dfHeadlines.iloc[[7589]])

# print(dfTweets.iloc[['2019-11-23']])
# Drop Unix column in Bitcoin price Dataframe since it's not needed
bitcoinPrices.drop(columns='unix', inplace=True)
bitcoinPrices.rename(columns={'date':'timestamp'}, inplace=True)
# Calculate log difference between opening and closing btc price for each day
bitcoinPrices['log_diff'] = np.log(bitcoinPrices['close']) - np.log(bitcoinPrices['open'])

# Add next day log price column
bitcoinPrices['next_day_log'] = bitcoinPrices.log_diff.shift(1)
bitcoinPrices['prev_day_log'] = bitcoinPrices.log_diff.shift(-1)
bitcoinPrices['trend'] = np.sign(bitcoinPrices['log_diff'])
bitcoinPrices['next_day_trend'] = np.sign(bitcoinPrices['next_day_log'])
bitcoinPrices.dropna(how='any', inplace=True)

# Convert Date columns for bitcoin, tweets, and news headlines to same date_time format
# This will allow for merging the dataframes by matching dates in each row
bitcoinPrices['timestamp'] = pd.to_datetime(bitcoinPrices['timestamp'])
bitcoinPrices['timestamp'] = bitcoinPrices['timestamp'].apply(lambda t: t.strftime('%Y-%m-%d'))

dfHeadlines['Date'] = pd.to_datetime(dfHeadlines['Date'])
dfHeadlines['Date'] = dfHeadlines['Date'].apply(lambda t: t.strftime('%Y-%m-%d'))

dfTweets['timestamp'] = pd.to_datetime(dfTweets['timestamp'])
dfTweets['timestamp'] = dfTweets['timestamp'].apply(lambda t: t.strftime('%Y-%m-%d'))
dfTweets = dfTweets.sort_values('timestamp')
# Reset index after sorting
dfTweets.reset_index(drop=True, inplace=True)
dfTweets.columns = ['timestamp', 'text']



# Remove newlines in tweet text
dfTweets.replace(r'\n', ' ', regex=True, inplace=True)
dfTweets.replace(r'\\n', ' ', regex=True, inplace=True)
dfTweets.replace(r'\r', ' ', regex=True, inplace=True)

# dfTweets.groupby(['timestamp'])['text\r'].apply(','.join).reset_index()
dfTweetsGrouped = dfTweets.groupby(['timestamp'])['text'].apply(','.join).reset_index()
dfTweets = dfTweetsGrouped
dfTweets.to_csv('Datasets/tweetsSentimentAnalysisSmall.csv', index=False,
                encoding='utf-8')

dfTweets[:-5]

tweets = dfTweets['text']


# dfTweets.groupby(['timestamp'])

# tweetTknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
# dfTweets['tokenized_tweet'] = dfTweets.apply(lambda row: tweetTknzr.tokenize(row['text']), axis=1)
# dfHeadlines['tokenized_headline'] = dfHeadlines.apply(lambda row: word_tokenize(row['headline']), axis=1)


# Conduct sentiment analysis using TextBlob package on tweets and then on headlines
# Define function to take input text and analyze sentiment
def calc_sentiment(txt):
    blob = TextBlob(txt)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

# calc sentiment polarity & subjectivity
tweets_sentiments = tweets.apply(calc_sentiment)
# Store polarity values in new column
tweets_polarity = tweets_sentiments.apply(lambda x: x[0])
# Store subjectivity values in new column
tweets_subjectivity = tweets_sentiments.apply(lambda x: x[1])

dfTweets['tweet_polarity'] = tweets_polarity
dfTweets['tweet_subjectivity'] = tweets_subjectivity  # create the series

dfTweets.sample(10)  # display 10 random rows

# Write tweets dataframe with polarity and subjectivity to new csv
# dfTweets[['timestamp', 'tweet_polarity', 'tweet_subjectivity', 'text']].to_csv('Datasets/tweetsSentimentAnalysisSmall.csv',
#                                                                    index=False,
#                                                                    encoding='utf-8')
dfHeadlines.columns = ['timestamp', 'headline']
# Group headlines comma delimited by date
dfHeadlinesGrouped = dfHeadlines.groupby(['timestamp'])['headline'].apply(','.join).reset_index()
dfHeadlines = dfHeadlinesGrouped

# Repeat sentiment analysis for news headlines
headlines = dfHeadlines['headline']

# Conduct sentiment analyis using TextBlob package on tweets and then on headlines
headline_sentiments = headlines.apply(
    calc_sentiment)  # calc sentiment polarity & subjectivity, return in a Series of tuples
headline_polarity = headline_sentiments.apply(lambda x: x[0])  # new column of polarity
headline_subjectivity = headline_sentiments.apply(lambda x: x[1])  # new column of subjectivity

dfHeadlines['news_polarity'] = headline_polarity
dfHeadlines['news_subjectivity'] = headline_subjectivity

dfHeadlines.sample(10)  # display 10 random rows

# Write tweets dataframe with polarity and subjectivity to new csv
# dfHeadlines.to_csv('Datasets/headlinesSentimentAnalysis.csv', index=False, encoding='utf-8')




# dfTweets = dfTweets[['timestamp', 'polarity', 'subjectivity', 'text']]
# groupedTweets = dfTweets.groupby(['timestamp', 'text'])[['polarity', 'subjectivity']].mean()
# # groupedTweets = dfTweets.groupby(['timestamp'])[str(dfTweets['polarity'])].apply(','.join).reset_index()
# groupedTweets[:-5]
# groupedTweets['polarity', 'subjectivity']
# groupedTweets.to_csv('Datasets/tweetsSentimentAnalysisGrouped.csv', index=False,
#                      encoding='utf-8')


# Sort values by date to prepare for merging
dfTweets = dfTweets.sort_values('timestamp')
dfHeadlines = dfHeadlines.sort_values('timestamp')
bitcoinPrices = bitcoinPrices.sort_values('timestamp')
# Merge sentiment scores from tweets and news headlines with bitcoin price dataframe
dfMerged = pd.DataFrame()
dfMerged = pd.merge(bitcoinPrices, dfHeadlines[['timestamp', 'news_polarity', 'news_subjectivity', 'headline']], on='timestamp', how='outer')
dfMerged = pd.merge(dfMerged, dfTweets[['timestamp', 'tweet_polarity', 'tweet_subjectivity', 'text']], on='timestamp', how='outer')

dfMerged.dropna(how='any', inplace=True)
# Sort merged df by date
dfMerged = dfMerged.sort_values('timestamp')
# Reset Index
dfMerged.reset_index(drop=True, inplace=True)

dfMergedClean = dfMerged
# Remove columns with headlines and tweets
dfMergedClean = dfMergedClean.drop(['text','headline'], axis = 1)
corrMatrix = dfMergedClean.corr()
print (corrMatrix)

import numpy as np
from sklearn.linear_model import LinearRegression

# Predict BTC Volume with all sentiment results as predictor variables
x = np.array(dfMerged[['news_polarity', 'news_subjectivity', 'tweet_polarity', 'tweet_subjectivity']]).reshape((-1, 4)) # coefficient of determination: 0.14175362009736936

# Predict BTC Volume with subjectivity results and then with polarity
x = np.array(dfMerged[['news_subjectivity', 'tweet_subjectivity']]).reshape((-1, 2)) # coefficient of determination: 0.1251332357454148
x = np.array(dfMerged[['news_polarity', 'tweet_polarity']]).reshape((-1, 2)) # coefficient of determination: 0.0850508771200924

# y = np.array(dfMerged['trend'])
y = np.array(dfMerged['Volume BTC'])

model = LinearRegression()
model.fit(x,y)
model = LinearRegression().fit(x, y)

r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)


# Predict trend with all sentiment results as predictor variables
x = np.array(dfMerged[['news_polarity', 'news_subjectivity', 'tweet_polarity', 'tweet_subjectivity']]).reshape((-1, 4)) # coefficient of determination: 0.014112806389795174
# y = np.array(dfMerged['trend'])
y = np.array(dfMerged['trend'])

model = LinearRegression()
model.fit(x,y)
model = LinearRegression().fit(x, y)

r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)


# Predict BTC Price trend with all sentiment results as predictor variables
x = np.array(dfMerged[['Volume BTC']]).reshape((-1, 1)) # coefficient of determination: 0.00893998336673818
# y = np.array(dfMerged['trend'])
y = np.array(dfMerged['trend'])

model = LinearRegression(normalize=True)
model.fit(x,y)
model = LinearRegression().fit(x, y)

r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)

dfModelData = dfMerged
dfModelData = dfMerged[['next_day_trend', 'trend', 'tweet_polarity', 'tweet_subjectivity', 'news_polarity', 'news_subjectivity']]
# Drop any rows where trend equals 0 (likely from price input error or missing price change in data)
dfModelData= dfModelData.loc[(dfModelData[['next_day_trend', 'trend']] != 0).all(axis=1)]
# Use numpy to convert to arrays
import numpy as np
# Labels are the values we want to predict
labels = np.array(dfModelData['trend'])
# labels = np.array(dfModelData['next_day_trend'])
# Remove the labels from the features
# axis 1 refers to the columns
features = dfModelData.drop(['next_day_trend', 'trend'], axis = 1)
# Saving feature names for later use
feature_list = list(features.columns)
# Convert to numpy array
features = np.array(features)

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.2, random_state = 42)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(train_features, train_labels);

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)
# Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

# Training Features Shape: (273, 4)
# Training Labels Shape: (273,)
# Testing Features Shape: (69, 4)
# Testing Labels Shape: (69,)
# Mean Absolute Error: 0.93 degrees.
# Accuracy: 85.42 %.

# Models for price trend prediction
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import precision_score
#
# # Finally, we perform ML and see results
# rf = RandomForestRegressor(n_estimators = 1000, random_state=0)
# rf.fit(train_features, train_labels);
# y_pred = rf.predict(test_features)
# df_res = pd.DataFrame({'y_test':test_features[:, 0], 'y_pred':y_pred})
# threshold = 0.5
# preds = [1 if val > threshold else 0 for val in df_res['y_pred']]
# # print(metrics.confusion_matrix(preds, df_res['y_test']))
# print('Accuracy Score:')
# print(accuracy_score(preds, df_res['y_test']))
# print('Precision Score:')
# print(precision_score(preds, df_res['y_test']))