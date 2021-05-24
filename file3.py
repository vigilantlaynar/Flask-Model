import tweepy
import time
import pandas as pd
pd.set_option('display.max_colwidth', 2000)


api_key = 'v2wdCQldpY41aInSKpFUTJ5bn'
api_secret_key ='j4qofzhUo2X7lxQ4PHbWxePwusTV4FXhgZEiOBTi9u9UnWlPDR'
access_token ='1295930167928344577-Yn6sSI6mr3S0MA3mjaPcmwHZnbeWmB'
access_token_secret ='TiXyuIoy9hND4GpOWBUnllcN1msF3WGYV5gBQEhuVIm9c'


authentication = tweepy.OAuthHandler(api_key, api_secret_key)
authentication.set_access_token(access_token, access_token_secret)


api = tweepy.API(authentication, wait_on_rate_limit=True)

def get_related_tweets(text_query):

    tweets_list = []

    count = 10
    try:

        for tweet in api.search(q=text_query, count=count):
            print(tweet.text)

            tweets_list.append({'created_at': tweet.created_at,
                                'tweet_id': tweet.id,
                                'tweet_text': tweet.text})
        return pd.DataFrame.from_dict(tweets_list)

    except BaseException as e:
        print('failed on_status,', str(e))
        time.sleep(3)
