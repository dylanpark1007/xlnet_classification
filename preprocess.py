

with open('/hdd/user4/xlnet_classification/dataset/short text/ag_news.test','r') as f:
    t = f.readlines()

data_index = []
data_text = []
data_label = []

for idx, line in enumerate(t):
    split_line = line.split(',')
    label_str = split_line[0]
    label = label_str[-2]
    text = split_line[1::]
    text = ('').join(text)
    data_index.append(idx)
    data_text.append(text)
    data_label.append(label)


import pandas as pd

df = pd.DataFrame(data=data_index , columns = ['index'])
df['text']=data_text
df['label'] = data_label

df.to_csv('/hdd/user4/xlnet_classification/dataset/short text/ag_news_test.csv')

