import csv
from matplotlib import pyplot as plt
with open('statistics.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter= ' ', quotechar='|')
    for row in spamreader:
        values = row[0].split(',')
        if values[-1] != '#' and values[-1] != '':
            print(str(values[-2])+'   '+str(values[-1]))
            print(float(values[-2]))
stat_result = [1,2,3,4,5,6,7,8,9,10]
plt.plot([i+1 for i in range(10)], stat_result)
plt.show()
