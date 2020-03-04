import pandas as pd


data = pd.read_csv('Data/news category classification.csv')

# to display the entire text
pd.set_option('display.max_colwidth', -1)


print(data.groupby(by='CATEGORY').size())

sample = data[data['CATEGORY'] == 'm']['TITLE']
print(sample[400:410])

