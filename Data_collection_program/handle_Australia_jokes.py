import csv
import re, string
count = 0

# used to process australia joke from twitter
with open('Austil2.csv', mode = 'r') as csv_file:
    count = 1609
    print(count)
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        joke = row['Embedded_text']
        count += 1
        joke = joke.strip()
        joke = joke.split('\n')[0]
        print(joke)
        with open('Austil' + str(count) + '.txt', 'w') as f:
            f.write(joke)
        # joke = re.sub(r'\d+$', '', joke)
        # print(joke)
        print(type(joke))
    
    print(count)