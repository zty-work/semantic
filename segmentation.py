import jieba
import jieba.posseg as pseg
import jieba.analyse
from collections import Counter

infile = open("./train_data/train.txt", "r")
outfile = open("./train_data/train_format.txt", "w")
outfile2 = open("./train_data/dic.txt", "w")
voc = []
while 1:
    line = infile.readline()
    if not line:
        break
    pass
    result = jieba.cut(line)
    words = [x for x in jieba.cut(line)]
    for i in words:
        voc.append(i)
    for t in result:
        outfile.write('<' + t)

c = Counter(voc).most_common(2000)
outfile2.write(str(c))

