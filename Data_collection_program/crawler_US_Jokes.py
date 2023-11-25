import re
from bs4 import BeautifulSoup
import requests

# used to crawler us jokes from jokes4us
url = 'http://www.jokes4us.com/miscellaneousjokes/worldjokes/index.html'
headers = {
  'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36 QIHU 360SE'
}
f = requests.get(url, headers = headers)
jokes_list = []
soup = BeautifulSoup(f.content, 'html.parser')
for link in soup.findAll('a', attrs={'href': re.compile("^http://www.jokes4us.com/miscellaneousjokes/worldjokes")}):
    # print(link.get('href'))
    jokes_list.append(link.get('href'))

jokes_list = jokes_list[27:]
count = 0
for anchor in jokes_list:
    joke_f = requests.get(anchor, headers=headers)
    joke_soup = BeautifulSoup(joke_f.content, 'html.parser')

    joke_content = joke_soup.find('div', {'class': 'LeftContent'}).p
    joke_content = joke_content.text
    lines = [i for i in joke_content.split('\n')]
    for i in range(len(lines)):
      if lines[i].startswith('Q:'):
        count += 1
        with open('usjokes' + str(count) + '.txt', 'w') as f:
          f.write(lines[i] + "\n")
          i += 1
          f.write(lines[i])
