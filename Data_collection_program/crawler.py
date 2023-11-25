from Scweet.scweet import scrape
import csv

# use Scweet library to crawler jokes from twitter
data = scrape(words=[], since="2020-11-16", until="2022-03-25", from_account="BadBritishJokes",
            interval=1,
            headless=False, display_type="Latest", save_images=False, proxy=None, save_dir='outputs',
            resume=False, filter_replies=True, proximity=False)


# save jokes from csv to txt files
csv_reader = csv.reader(open("./BadBritishJokes_2020-11-16_2022-03-25.csv"))
flag = 0
count = 4039
for line in csv_reader:
    if flag == 0:
        flag = 1
        continue
    with open('./British/' + str(count) + ".txt", 'w') as f:
        f.write(line[4])
    count += 1
