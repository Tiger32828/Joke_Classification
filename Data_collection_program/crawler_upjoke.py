import re
from bs4 import BeautifulSoup
import requests

def removeSGML(raw_string):
  cleaner = re.compile('<.*?>') 
  clean_string = re.sub(cleaner, '', raw_string)
  return clean_string


url = 'https://upjoke.com/china-jokes'
headers = {
  'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36 QIHU 360SE'
}
f = requests.get(url, headers = headers)
jokes_list = []
soup = BeautifulSoup(f.content, 'html.parser')
joke_content = soup.findAll('div', {'class': 'joke-body'})
count = 0
for joke in joke_content:
    with open('./Chinese/' + str(count) + ".txt", 'w') as f:
        f.write(removeSGML(str(joke)))
    count += 1