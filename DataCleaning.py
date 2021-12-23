#Read Both the files
f=open("Data2/english.txt", encoding="utf8")
data1 = f.read().split('\n')
f.close()
f=open("Data2/urdu.txt", encoding="utf8")
data2 = f.read().split('\n')
f.close()
print(len(data1),len(data2))
'''
import re
def clean_data(text):
  #text = re.sub(r"[؟,'?\/.$%_():!۔،’‘#-;<>@$%^&*👍🏾—]", " ", text, flags=re.I)
  #text = re.sub(r'"', " ", text, flags=re.I)
  #text = re.sub(r"[a-zA-Z]"," ",text,flags=re.I)
  #text = re.sub(r"\d", " ", text)
  #text = re.sub(r"[\n×\t”“£]", " ", text)
  #return re.findall("[a-zA-Z]+",text)
  return re.findall("[\x]+",text)
  '''


