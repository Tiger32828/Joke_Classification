import re
# remove hash tag
for i in range(0,9252):
    with open("./Spanish_trans/" + str(i) + ".txt", "r") as f:
        text = f.read()
        res = ' '.join(re.sub("(#[A-Za-z0-9]+)","",text).split())
        f1 = open("./Spanish_pro/" + str(i) + ".txt",'w')
        f1.write(res)