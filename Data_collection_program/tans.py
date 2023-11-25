from deep_translator import (GoogleTranslator)

# translate joke from chinese to english
for i in range(0,3364):
    with open("./Chinese/" + str(i) + ".txt", "r") as f:
        text = f.read()
        translated = GoogleTranslator(source='auto', target='en').translate(text=text)
        f1 = open("./Chinese_trans/" + str(i) + ".txt",'w')
        f1.write(translated)