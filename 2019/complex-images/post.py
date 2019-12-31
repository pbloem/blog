"""
Post the image final.png to the tony ants twitter account.

"""

import requests, sys, os, time, yaml
from requests_oauthlib import OAuth1

def check(r):
    if r.status_code < 200 or r.status_code > 299:
        print(r.status_code, r.reason)
        print(r.text)
        sys.exit()


# Creds for @ants_tiny
creds = yaml.load(open('auth.yaml', 'r'))

auth = OAuth1(
    creds['app_key'], # app key
    creds['app_secret'], # app secret
    creds['user_key'], # user key
    creds['user_secret'] # user secret
)

# Upload the image (chunked upload)

URL = 'https://upload.twitter.com/1.1/media/upload.json'

if os.path.exists('final.gif'):
    filename = 'final.gif'
    mime = 'image/gif'
    gif = True
else:
    filename = 'final.png'
    mime = 'image/png'
    gif = False

file = open(filename, 'rb')
total_bytes = os.path.getsize(filename)

## INIT

data = {
      'command': 'INIT',
      'media_type': mime,
      'total_bytes': total_bytes,
      'media_category': 'tweet_gif' if gif else 'tweet_image'
    }

r = requests.post(URL, data=data, auth=auth)
media_id = r.json()['media_id']

# APPEND

bytes_sent = 0
segment_id = 0
while bytes_sent < total_bytes:
    print('appending')

    data = {
        'command': 'APPEND',
        'media_id': media_id,
        'segment_index': segment_id
    }

    files = {
        'media': file.read(4 * 1024 * 1024)
    }

    r = requests.post(url=URL, data=data, files=files, auth=auth)
    check(r)

    segment_id += 1
    bytes_sent = file.tell()

    print(f'{bytes_sent/1024:.5} of {total_bytes/1024:.5} kbytes uploaded')

# FNALIZE

data = {
  'command': 'FINALIZE',
  'media_id': media_id
}

print('finalizing')
r = requests.post(url=URL, data=data, auth=auth)
check(r)
print('FINALIZE', r.json())

# Post the tweet
URL = 'https://api.twitter.com/1.1/statuses/update.json'
data = {'status' : f'', 'media_ids' : media_id}

r = requests.post(URL, data=data, auth=auth)
print(r.status_code, r.reason)
print(r.json())

