import pandas as pd 
import nest_asyncio
nest_asyncio.apply()
import twint
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer


sid = SentimentIntensityAnalyzer()

import re
import nltk
nltk.download('words')
words = set(nltk.corpus.words.words())

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

def main():
    ldf = list()
    cities = input('Enter names of cities: ').split(',')
    keywords = input('Enter keywords to search: ').split(',')
    Since = input('Enter FROM date YYYY-MM-DD: ')
    Util = input('Enter TO date YYYY-MM-DD: ')
    Limit = int(input('Enter max number of tweets:'))
    file_name = input('Enter file name: ')

    for city in cities:
        for keyword in keywords:
            c = twint.Config()
            c.Search = keyword
            c.Lang= "en"
            c.Pandas = True
            c.Since = Since
            c.Until = Util
            c.Near = city
            c.Limit =Limit
            twint.run.Search(c)
            tmp = twint.output.panda.Tweets_df


            ldf.append(tmp)
    data = pd.concat(ldf, ignore_index = True)


  # sort by date
    data.sort_values(by='date',ascending=False)

    # making boolean to find engish
    filter1 = data["language"]=="en"

    # filtering data
    df01 = data.where(filter1)

    # dropping null values
    df1 = df01.dropna()

    # making boolean to find arabic
    filter2 = data["language"]=="ar"

    # filtering data
    df02 = data.where(filter2)

    # dropping null values
    df2 = df02.dropna()

    frames = [df1, df2]

    result = pd.concat(frames)
    new_df = result.sort_values(by='date',ascending=False)


    def cleaner(tweet):
        tweet = re.sub("@[A-Za-z0-9]+","",tweet) #Remove @ sign
        tweet = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", tweet) #Remove http links
        tweet = " ".join(tweet.split())
        tweet = tweet.replace("#", "").replace("_", " ") #Remove hashtag sign but keep the text
        tweet = " ".join(w for w in nltk.wordpunct_tokenize(tweet)
             if w.lower() in words or not w.isalpha())
        return tweet


    new_df['tweet_clean'] = new_df['tweet'].apply(cleaner)



    sid = SentimentIntensityAnalyzer()
   #sid.lexicon.update(word_dict)

    list1 = []
    for i in new_df['tweet_clean']:
        list1.append((sid.polarity_scores(str(i)))['compound'])

    new_df['sentiment'] = pd.Series(list1)

    def sentiment_category(sentiment):
        label = ''
        if(sentiment>0):
            label = 'positive'
        elif(sentiment == 0):
            label = 'neutral'
        else:
            label = 'negative'
        return(label)

    new_df['sentiment_category'] = new_df['sentiment'].apply(sentiment_category)

    new_df.to_excel(file_name)

if __name__=="__main__":
    main()
    
