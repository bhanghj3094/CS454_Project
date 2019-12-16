import csv

# list to write
total = [[] for _ in range(30)]

for i in range(48):
  name = "logs_statistics_" + str(i) + "/log_cgp.txt"

  f = open(name, "r")
  lines = f.readlines()
  for j in range(len(lines)):
    line = lines[j] 
    list = line.split(",")
    total[j].append(list[3])
  f.close()

w = open("fitness.csv", 'a')
writer = csv.writer(w, lineterminator='\n') 
for i in range(30):
  writer.writerow(total[i]) 
w.close()

