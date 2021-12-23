#Read Both the files
f=open("data/Eng_train.txt", encoding="utf8")
data1 = f.read().split('\n')
f.close()
f=open("data/Urd_train.txt", encoding="utf8")
data2 = f.read().split('\n')
f.close()
print(len(data1),len(data2))


tf1 = open("data/Eng12.txt", "w",encoding="utf8")
tf2 = open("data/Urd12.txt", "w",encoding="utf8")
count=0
for i,j in zip(data1,data2):
  if(len(i.split())<12):
    count=count+1
    tf1.write(i + "\n")
    tf2.write(j + "\n")

tf1.close()
tf2.close()
print(count)

