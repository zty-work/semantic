infile1 = open("./train_data/train_format.txt", "r")
infile2 = open("./train_data/dic-for.txt", "r")
outfile = open("./train_data/training.txt", "w")
outfile2 = open("./train_data/testing.txt", "w")

matrix_yt = [0]
matrix_xt = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0,
             0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
matrix_y = [0]
matrix_x = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0,
            0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
print(len(matrix_xt))
print(len(matrix_x))
dict = []
while 1:
    line2 = infile2.readline()
    if not line2:
        break
    pass
    word = line2.split(",")
    for a in word:
        dict.append(a)
z = 0
while 1:
    line = infile1.readline()
    if not line:
        break
    if z >= 90000:
        groups = line.split("	")
        list1 = groups[0].split("<")
        list2 = groups[1].split("<")
        for i in range(len(list1)):
            for index in range(len(dict)):
                if list1[i] == dict[index]:
                    if i < 25:
                        matrix_xt[i] = index
        for j in range(len(list2)):
            for index in range(len(dict)):
                if list2[j] == dict[index]:
                    if j < 25:
                        matrix_xt[j + 25] = index
        if groups[2][1] == "1":
            matrix_yt[0] = 1
        outfile2.write(str(matrix_xt[0:25]))
        outfile2.write(".")
        outfile2.write(str(matrix_xt[25:]))
        outfile2.write(".")
        outfile2.write(str(matrix_yt))
        outfile2.write('\n')
        matrix_xt = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0,
                     0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        matrix_yt = [0]
    pass
    if z < 90000:
        z = z + 1
        groups = line.split("	")
        list1 = groups[0].split("<")
        list2 = groups[1].split("<")
        for i in range(len(list1)):
            for index in range(len(dict)):
                if list1[i] == dict[index]:
                    if i < 25:
                        matrix_x[i] = index
        for j in range(len(list2)):
            for index in range(len(dict)):
                if list2[j] == dict[index]:
                    if j < 25:
                        matrix_x[j + 25] = index
        if groups[2][1] == "1":
            matrix_y[0] = 1
        outfile.write(str(matrix_x[0:25]))
        outfile.write(".")
        outfile.write(str(matrix_x[25:]))
        outfile.write(".")
        outfile.write(str(matrix_y))
        outfile.write('\n')
        matrix_x = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0,
                    0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        matrix_y = [0]

