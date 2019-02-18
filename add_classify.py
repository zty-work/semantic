infile1 = open("./train_data/training.txt", "r")

outfile3 = open("./train_data/classes.txt", "w")
classOfSen = []


def search(tempWords1, tempWords2, classOfSen):
    for i in range(0, len(classOfSen)):
        for j in range(0, len(classOfSen[i])):
            if tempWords1 == classOfSen[i][j]:
                classOfSen[i].append(tempWords2)
                return 1, classOfSen
            if tempWords2 == classOfSen[i][j]:
                classOfSen[i].append(tempWords1)
                return 1, classOfSen
    temp = [tempWords1, tempWords2]
    classOfSen.append(temp)
    return 0, classOfSen


while 1:
    line1 = infile1.readline()
    if not line1:
        break
    pass
    trainingY = line1.split(".")[2]
    if trainingY == "[1]\n":
        tempWords1 = line1.split(".")[0].split(",")[1:-2]
        tempWords2 = line1.split(".")[1].split(",")[1:-2]
        flag, classOfSen = search(tempWords1, tempWords2, classOfSen)

# for i in range(0, len(classOfSen)):
#    for j in range(0,len(classOfSen[i])):
#        for k in range(0,len(classOfSen[i][j])):
#            outfile1.write(classOfSen[i][j][k])
#            if k!=len(classOfSen[i][j])-1:
#                outfile1.write(",")
#        if j!=len(classOfSen[i])-1:
#            outfile1.write(".")
#    outfile1.write('\n')

for i in range(0, len(classOfSen)):
    for j in range(0, len(classOfSen[i])):
        outfile3.write(str(i))
        outfile3.write(",")
        for k in range(0, len(classOfSen[i][j])):
            outfile3.write(classOfSen[i][j][k])
            if k!=len(classOfSen[i][j])-1:
                outfile3.write(",")
        outfile3.write('\n')

#outfile2.write(str(len(classOfSen)))
