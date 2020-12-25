'''
This is a crawler for crawling data from 'dpchallenge.com'.
Proxy is required because of the anti-crawling mechanism of this site, or your IP will be blocked.
Images will be saved under 'image_dir_path' and their info will be saved in 'cap_json_path'.
'''

import requests
import re
import sys, os
import json
from bs4 import BeautifulSoup
import hashlib
import time
import random

from PIL import Image
import warnings
import piexif
from tqdm import tqdm
import numpy as np

base_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

cap_json_path = os.path.join(base_path, 'raw.json')
image_dir_path = os.path.join(base_path, 'images')

baseurl = 'https://www.dpchallenge.com'
url = 'https://www.dpchallenge.com/photo_gallery.php?GALLERY_ID=40'

wait_second = 0.3  # waiting time before retry
time_out = 60
min_html_size = 1024 # used to judge if the ip is blocked by the site

# proxy ip
proxy_ip = 'secondtransfer.moguproxy.com:9001'
# proxy Authorization
appKey = ""
user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.132 Safari/537.36"
proxies = {
    "http": "http://" + proxy_ip,
    "https": "https://" + proxy_ip
}
headers = {
    "Proxy-Authorization": 'Basic '+ appKey,
    "User-Agent": user_agent,
}

# request a certain URL and return response
def getResponse(url):
    while True:
        try:
            # ignor "unverified https request" warning
            requests.packages.urllib3.disable_warnings()
            response = requests.get(
                url = url, 
                headers = headers,
                proxies = proxies,
                timeout = time_out,
                verify = False,
                allow_redirects = False,
            )
            if len(response.content) < min_html_size:
                # the proxy ip has been blocked, should change an ip
                print("proxy has been blocked: ", response.content)
            else:
                return response
        except Exception as e:
            print("proxy connection error: ",  str(e))
        finally:
            time.sleep(wait_second)


# get the number of pages
def getPageNum():
    response = getResponse(url)
    pageNum = re.findall(r'<small>\[(.*?)\]</small>', response.text)
    return int(pageNum[0])


# get image list of each page
def getImageList(page):
    pageURL = url + '&page=' + str(page)
    response = getResponse(pageURL)
    itemList = re.findall(r'width="20%"><a href="(.*?)" class="i"><img src="', response.text)
    return itemList


# get the "Statistics" data of each image
def getStat(soup):

    infoList = soup.find_all('tr', {'class':'forum-bg1'})[1].contents[1].contents  
    statData = {}

    for i in range(len(infoList)):

        if str(infoList[i]) == '<b>Avg (all users):</b>':
            statData['avg_all_users'] = str(infoList[i + 1]).strip()
            continue
        elif str(infoList[i]) == '<b>Avg (commenters):</b>':
            statData['avg_commenters'] = str(infoList[i + 1]).strip()
            continue
        elif str(infoList[i]) == '<b>Avg (participants):</b>':
            statData['avg_participants'] = str(infoList[i + 1]).strip()
            continue
        elif str(infoList[i]) == '<b>Avg (non-participants):</b>':
            statData['avg_non_participants'] = str(infoList[i + 1]).strip()
            continue
        elif str(infoList[i]) == '<b>Views since voting:</b>':
            statData['views_since_voting'] = str(infoList[i + 1]).strip()
            continue
        elif str(infoList[i]) == '<b>Views during voting:</b>':
            statData['views_during_voting'] = str(infoList[i + 1]).strip()
            continue
        elif str(infoList[i]) == '<b>Votes:</b>':
            statData['votes'] = str(infoList[i + 1]).strip()
            continue
        elif str(infoList[i]) == '<b>Comments:</b>':
            statData['comments_num'] = str(infoList[i + 1]).strip()
            continue

    return statData


# get all comments of each image
def getComments(soup):
    comments = []
    for item in soup.find_all('table', {'class', 'forum-post'}):
        comments.append(item.contents[0].contents[0].get_text().strip())
    return comments


# download an image
def getImage(imgURL, imgID):
    img_path = os.path.join(image_dir_path, imgID)
    response = getResponse(imgURL)
    with open(img_path, "wb") as f:
        f.write(response.content)


# find out corrupt images and fix them
def checkImage():
    # read data from 'cap_json_path'
    with open(cap_json_path, 'r', encoding = 'utf-8') as f:
        data = json.load(f)
        f.close()

    print("Checking and fixing images...")

    # recognize 'UserWarning' as error, for finding images with corrupt EXIF info
    warnings.filterwarnings("error", category = UserWarning)

    for cnt, imgID in enumerate(tqdm(data, desc = 'Checked images: ')):
        image_path = os.path.join(image_dir_path, imgID)
        while True:
            try:
                # verify that it is in fact an validate image
                img = Image.open(image_path).resize((256,256))
            # image corrupt
            except (IOError, SyntaxError, ValueError) as e:
                print('Bad image:', imgID)
                # download it again
                os.remove(image_path)
                getImage(data[imgID]['image_url'], imgID)
            # EXIF info corrupt, remove EXIF info
            except warning as e:
                print('Bad EXIF info:', imgID)
                piexif.remove(image_path)
            else:
                # validate image, go to the next
                break


if __name__ == '__main__':

    pageNum = getPageNum()
    print("there are ", pageNum, " pages")

    # read file 'cap_json_path'
    with open(cap_json_path, 'r') as f:
        data = json.load(f)
        f.close()
    
    for page in range(171, pageNum):

        print("--------- page ", page, " start ---------")

        itemList = getImageList(page)
        
        for item in itemList:

            # image ID
            imgID = item.lstrip('/image.php?IMAGE_ID=') + '.jpg'
            
            # ignore the images that are already exist
            if imgID in data:
                print("image ", imgID, " is already exist")
                continue

            imgData = {}

            itemURL = baseurl + item
            itemHTML = getResponse(itemURL).text

            soup = BeautifulSoup(itemHTML, "lxml")


            # basic info
            imgURL = 'http:' + soup.find_all('img', {'style':'border: 1px solid black'})[0]['src']
            imgData['image_url'] = imgURL
            imgData['date_uploaded'] = re.findall(r'<b>Date Uploaded:</b>(.*?)<br>', itemHTML)[0].strip()


            # comments
            comments = getComments(soup)
            if comments == []:
                # no comments, ignore
                continue
            else:
                imgData['comments'] = comments

            # "Statistics" info
            if any(re.findall(r'<td nowrap>Statistics</td>', itemHTML)):
                imgData.update(getStat(soup))
                data[imgID] = imgData
            else:
                # no "Statistics" info, ignore
                continue
            
            print("--- getting image ", imgID, " ... ---")

            # download image
            getImage(imgURL, imgID)

            # write the image data into file 'cap_json_path'
            with open(cap_json_path, 'w') as f:
                json.dump(data, f)
                f.close()

            time.sleep(random.randint(3, 10))
        
        print("--------- page ", page, " done ---------")
        time.sleep(random.randint(5, 20))
    
    # find out corrupt images and fix them
    checkImage()