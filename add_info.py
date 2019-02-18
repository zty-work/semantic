infile1 = open("./train_data/training.txt", "r")
infile2 = open("./train_data/testing.txt", "r")
outfile1 = open("./train_data/training_info.txt", "w")
outfile2=open("./train_data/testing_info.txt", "w")
def search(tempWord, wordOfSen):
    for word in wordOfSen:
        if tempWord == word:
            return 1
    return 0


while 1:
    info1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    info2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    line1 = infile1.readline()
    if not line1:
        break
    pass
    wordOfSen1 = line1.split(".")[0].split(",")[1:-2]
    wordOfSen2 = line1.split(".")[1].split(",")[1:-2]
    for i in range(0, len(wordOfSen1)):
        for j in range(0, len(wordOfSen2)):
            if (wordOfSen1[i] == wordOfSen2[j]) & (wordOfSen1[i] != ' 0') & (wordOfSen1[i] != ' 607'):
                info1[i] = 1
                info2[j] = 1
    outfile1.write(wordOfSen1[0])
    for a in range(1,len(wordOfSen1)):
        outfile1.write(",")
        outfile1.write(wordOfSen1[a])
    outfile1.write(".")
    outfile1.write(" ")
    outfile1.write(str(info1[0]))
    for c in range(1, len(info1)):
        outfile1.write(", ")
        outfile1.write(str(info1[c]))
    outfile1.write(".")
    outfile1.write(wordOfSen2[0])
    for a in range(1, len(wordOfSen2)):
        outfile1.write(",")
        outfile1.write(wordOfSen2[a])
    outfile1.write(".")
    outfile1.write(" ")
    outfile1.write(str(info2[0]))
    for d in range(1, len(info2)):
        outfile1.write(", ")
        outfile1.write(str(info2[d]))
    outfile1.write(".")
    outfile1.write(line1.split(".")[-1])

while 1:
    info1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    info2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    print(len(info1))
    line2 = infile2.readline()
    if not line2:
        break
    pass
    wordOfSen1 = line2.split(".")[0].split(",")[1:-2]
    wordOfSen2 = line2.split(".")[1].split(",")[1:-2]
    for i in range(0, len(wordOfSen1)):
        for j in range(0, len(wordOfSen2)):
            if (wordOfSen1[i] == wordOfSen2[j]) & (wordOfSen1[i] != ' 0') & (wordOfSen1[i] != ' 607'):
                info1[i] = 1
                info2[j] = 1
    outfile2.write(wordOfSen1[0])
    for a in range(1,len(wordOfSen1)):
        outfile2.write(",")
        outfile2.write(wordOfSen1[a])
    outfile2.write(".")
    outfile2.write(" ")
    outfile2.write(str(info1[0]))
    for c in range(1, len(info1)):
        outfile2.write(", ")
        outfile2.write(str(info1[c]))
    outfile2.write(".")
    outfile2.write(wordOfSen2[0])
    for a in range(1, len(wordOfSen2)):
        outfile2.write(",")
        outfile2.write(wordOfSen2[a])
    outfile2.write(".")
    outfile2.write(" ")
    outfile2.write(str(info2[0]))
    for d in range(1, len(info2)):
        outfile2.write(", ")
        outfile2.write(str(info2[d]))
    outfile2.write(".")
    outfile2.write(line1.split(".")[-1])