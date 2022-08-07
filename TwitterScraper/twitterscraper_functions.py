import snscrape.modules.twitter as sntwitter
import pandas as pd


def get_tweets(name, since, until):
    # function to get tweets by username, starting date and end date
    tweet_list = []
    for i, tweet in enumerate(
            sntwitter.TwitterSearchScraper(
                'from:' + name + ' since_time:' + since + ' until_time:' + until).get_items()):
        tweet_list.append(
            [tweet.date, tweet.id, tweet.user.username, tweet.user.followersCount, tweet.likeCount, tweet.retweetCount
                , tweet.content])  # declare the attributes to be returned

    # Creating a dataframe from the tweets list above
    tweets_df1 = pd.DataFrame(tweet_list, columns=['Datetime', 'Tweet Id', 'Username', 'FollowersCount', 'LikeCount',
                                                   'RetweetCount', 'Text'])
    return tweets_df1


def get_keyword_tweets(keyword, since, until,maxTweets):
    # function to get tweets by username, starting date and end date
    tweet_list = []

    for i, tweet in enumerate(sntwitter.TwitterSearchScraper( keyword + ' since_time:' + since + ' until_time:' + until).get_items()):
        if i > maxTweets:
            break
        tweet_list.append([tweet.date, tweet.content])  # declare the attributes to be returned

    # Creating a dataframe from the tweets list above
    tweets_df1 = pd.DataFrame(tweet_list, columns=['Datetime', 'Text'])
    return tweets_df1
