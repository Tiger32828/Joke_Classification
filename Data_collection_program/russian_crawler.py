# This is the crawler used for retriving Russian jokes from https://www.anekdot.ru
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import sys
# The translator package
from deep_translator import GoogleTranslator

def main():
    queue = []
    # counter for jokes retrived
    count = 1
    seed = "https://www.anekdot.ru/release/anekdot/day/"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36'}
    queue.append(seed)
    while queue:
        stop = False
        url = queue[0]
        queue.pop()
        # see if the page can be retrived in a short time
        try:
            result = requests.get(url, headers=headers, timeout=1)
        except:
            continue
        # retrive only HTML pages
        if "text/html" not in result.headers["content-type"]:
            continue
        # pharse the page
        soup = BeautifulSoup(result.content, 'lxml')
        # get certain tags
        for text in soup.find_all(lambda tag: tag.name == 'div' and tag.get('class') == ['topicbox']):
            if_text = False
            f = open("russian" + str(count), 'w', encoding='utf-8')
            # get text within tags
            for i in text.find_all(lambda tag: tag.name == 'div' and tag.get('class') == ['text']):
                ru_text = i.get_text()
                # prevent the program from API connection error
                try:
                    # translate
                    eng_text = GoogleTranslator(source='russian', target='english').translate(text=ru_text)
                except:
                    continue
                f.write(eng_text)
                if_text = True
            if if_text:
                count += 1
            # retrive 6000 jokes
            if (count > 6000):
                stop = True
                break
        if stop == True:
            break
        # get the next URL
        for link in soup.find_all(lambda tag: tag.name == 'div' and tag.get('class') == ['voteresult']):
            get_link = link.find('a').get('href')
            queue.append("https://www.anekdot.ru"+str(get_link))





if __name__ == '__main__':
    main()