from deep_translator import GoogleTranslator
translator = GoogleTranslator(source='en', target='ur')

f=open("/content/drive/MyDrive/FYP/MonoData/English/e10.txt", encoding="utf8")
data = f.read().split('\n')
f.close()
f=open("/content/drive/MyDrive/FYP/MonoData/English/u10.txt", encoding="utf8")
data2 = f.read().split('\n')
f.close()

stIdx=len(data2)-1
EndIdx=len(data)


with open("/content/drive/MyDrive/FYP/MonoData/English/u10.txt", "a") as textfile:
  for i in range(stIdx,EndIdx):
    textfile.write(translator.translate(data[i].lower())+"\n")
    if (i%100==0):
      print(i)
print("Done")

#arr=np.array(data1)
#np.savetxt("/content/drive/MyDrive/FYP/MonoData/English/e1.txt",arr,fmt="%s",encoding="utf8")

