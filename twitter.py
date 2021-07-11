#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""Twitter API module:
Enables Twitter streaming and eventually supports cursor data retrieval"""

import json
import os
import tweepy as tw

from pprint import pprint
from sys import exit
from typing import Union


def parse_twitter_credentials(config_file: str) -> str:
    """Parse twitter credential file to retrieve account credentials
    
    Arguments:
        config_file {str} -- Filepath to config file
    
    Returns:
        str -- account credentials
    """
    with open(config_file, 'r') as c:
        config = dict()
        for conf in c.readlines():
            conf = conf.split('=')
            config[conf[0]] = conf[1].strip()

    return config['API key'], config['API secret key'], config[
        'Access token'], config['Access token secret']


def authentication(consumer_key: str, consumer_secret: str, access_token: str,
                   access_token_secret: str):
    """Authentification for twitter api
    
    Arguments:
        consumer_key {str} -- Credentials of Twitter account 
        consumer_secret {str} -- Credentials of Twitter account
        access_token {str} -- Credentials of Twitter account
        access_token_secret {str} -- Credentials of Twitter account
    
    Returns:
        [obj] -- twitter api access
    """
    auth = tw.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    # twitter states a limit of 900 tweets per 15 minutes
    # authenticate for 5000 tweets and split up in batches /w re-auth
    return tw.API(auth,
                  wait_on_rate_limit=True,
                  wait_on_rate_limit_notify=True,
                  parser=tw.parsers.JSONParser())


def parse_json_config(search_config_json: str,
                      config_type: str = 'streaming') -> dict:
    """Parse search config for twitter stream

    Arguments:
        search_config {str} -- Filepath to config json file
        config_type {str} -- Parses config for one specific configuration object
    
    Returns:
        dict -- Dictionary of configs to pass into filter_stream

    Config in JSON format as array of possible settings-objects which are a combination of parameter & boolean to indicate usage.
    Some parameters must be given as arguments in `stream.filter()`, while other parameters like language are used in `get_status()` to pre-filter and save storage.
    Example:
    ```json
    {
        "streaming_config": {
            "lang": "en",
            "tracks": ["corona", "china"]
        },
        "filter_config": {
            "text": true,
            "user": {
            "location": true
            },
            "place": true,
            "entities": {
            "hashtags": {
                "text": true
            }
            },
            "favourites_count": true,
            "retweet_count": true
        }
    }
    ```
    """
    with open(search_config_json, 'r') as f:
        CFG = json.load(f)
    if config_type == 'streaming':
        return CFG['streaming_config']
    else:
        return CFG['filter_config']


# * STREAM
class Streaming(tw.StreamListener):
    """Streaming class for Twitter"""
    http_error_codes = {
        '200': True,
        '401': False,
        '403': False,
        '404': True,
        '406': True,
        '413': False,
        '416': False,
        '420': False,
        '503': False
    }

    def __init__(self, filepath):
        """Initialize Streaming Listener & Config
        
        Arguments:
            filepath {str} -- filepath to store tweets in
        """
        super(Streaming, self).__init__()
        self._CFG = None
        self.FILEPATH = filepath
        self.create_config()

    def create_config(self) -> dict:
        """Creates configuration for streaming filter, as language or keywords
        
        Returns:
            dict -- decoded JSON configuration file
        """
        try:
            SEARCH_CONFIG = './twitter_config.json'
            self._CFG = parse_json_config(SEARCH_CONFIG,
                                          config_type='streaming')
        except FileNotFoundError:
            SEARCH_CONFIG = os.getcwd() + '/configs/twitter_config.json'
            self._CFG = parse_json_config(SEARCH_CONFIG,
                                          config_type='streaming')
        print('Use following configuration:\n')
        pprint(self._CFG)

    def on_status(self, status):
        # override to store tweets
        # printing status only returns full twitter object
        # twitter objects https://developer.twitter.com/en/docs/tweets/data-dictionary/overview/tweet-object
        # access twitter objects with dot notation
        with open(self.FILEPATH, 'a') as tweets:
            json.dump(status._json, tweets)
            tweets.write('\n')

    def on_error(self, status_code) -> Union[bool, str]:
        """If http response returns false, stream client will disconnect,
        if http response returns true, it reconnects
        
        Arguments:
            status_code {str} -- http response status code
        
        Returns: Union[bool, str] -- status code and error message
        """
        # status codes: https://developer.twitter.com/en/docs/tweets/filter-realtime/guides/connecting
        # return False to disconnect
        # return True to reconnect
        try:
            if status_code in self.http_error_codes.keys():
                return self.http_error_codes[status_code]
        except KeyError:
            return f'Unknown http response: {status_code}'


class Cursor():
    pass
    # raise NotImplementedError("may be implemented later")


if __name__ == "__main__":
    #! authentification
    try:
        CREDENTIALS_FILE = './credentials.txt'
        api_key, api_secret, access_token, access_token_secret = parse_twitter_credentials(
            CREDENTIALS_FILE)
    except FileNotFoundError:
        CREDENTIALS_FILE = os.getcwd() + '/configs/credentials.txt'
        api_key, api_secret, access_token, access_token_secret = parse_twitter_credentials(
            CREDENTIALS_FILE)

    api = authentication(api_key, api_secret, access_token,
                         access_token_secret)
    try:
        api.verify_credentials()
        print('Authentification ok')
    except:
        print('Could not authenticate with Twitter.\nExiting...')
        sys.exit(0)

    #! TESTING AREA

    stream_listener = Streaming('./tweets.txt')
    stream = tw.Stream(auth=api.auth, listener=stream_listener)
    stream.filter(**stream_listener._CFG)