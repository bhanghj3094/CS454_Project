import math
import csv
from matplotlib import pyplot as plt
def normal_cdf(x, mu=0, sigma=1):
    return ((1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2)


def normal_cdf_diff(x, mu, sigma, n):
    return (((1 + math.erf((x +0.00000001 - mu) / math.sqrt(2) / sigma)) / 2)**n
            - ((1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2)**n) * 100000000

def exp(mu,sigma,n):
    result = 0
    ran = 1000000
    int_range = [x / 10000.0 for x in range(-ran,ran)]
    

    for i in int_range:
        result += i*normal_cdf_diff(i,mu,sigma,n)/10000.0

    int_range_1 = [x / 1000.0 for x in range(800,1500)]
    #plt.plot(int_range_1,[normal_cdf_diff(i,mu,sigma,n) for i in int_range_1],'-',label='mu=0,sigma=1')
    print (result)
    return result


generation = 1
stat_result = []
with open('statistics.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter= ' ', quotechar='|')
    for row in spamreader:
        values = row[0].split(',')
        if values[-1] != '#' and values[-1] != '':
            mean = float(values[-2])
            std = float(values[-1])
            if generation in [1,2,3,4,5,10,20,30]:
                print(str(generation) + ': ')
            stat_result.append(exp(mu = mean, sigma = std, n = 30/generation))
            generation += 1
plt.plot([i+1 for i in range(30)], stat_result)
plt.xlabel('generation')
plt.ylabel('E(X) of Fitness(Accuracy)')
plt.show()
