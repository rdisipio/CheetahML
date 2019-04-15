#!/usr/bin/env python3
# -*- coding: utf-8 -*-
## run
## > python flickr_GetUrl.py tag number_of_images_to_attempt_to_download
from flickrapi import FlickrAPI
import pandas as pd
import sys
key='6d43c0447395f3d14fef801ee950cf20'
secret='e390ead002e4bdad'

def get_urls(image_tag,MAX_COUNT):
    flickr = FlickrAPI(key, secret)
    photos = flickr.walk(text=image_tag,
                            tag_mode='all',
                            tags=image_tag,
                            extras='url_s',
                            per_page=100,
                            sort='relevance')
    count=0
    urls=[]
    for photo in photos:
        if count< MAX_COUNT:
            print("Fetching url for image number {}".format(count))
            try:
                url=photo.get('url_s').strip()
                if url == "": continue
                urls.append(url)
            except:
                print("Url for image number {} could not be fetched".format(count))
            count=count+1

        else:
            print("Done fetching urls, fetched {} urls out of {}".format(len(urls),MAX_COUNT))
            break
    urls=pd.Series(urls)
    print("Writing out the urls in the current directory")
    urls.to_csv(image_tag+"_urls.csv")
    print("Done!!!")
def main():
    tag=sys.argv[1]
    MAX_COUNT=int(sys.argv[2])
    get_urls(tag,MAX_COUNT)
if __name__=='__main__':
    main()

