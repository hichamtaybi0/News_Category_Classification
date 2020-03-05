# Import Libraries
import pandas as pd  # CSV file I/O (pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

news = pd.read_json('Data/News_Category_Dataset_v2.json', lines=True)
# remove_columns_list = ['authors', 'date', 'link', 'short_description', 'headline']
# combine headline + short_description into text
news['text'] = news[['headline', 'short_description']].apply(lambda x: ' '.join(x), axis=1)

# print(news.shape)
# print(type(news))

# To display entire text
pd.set_option('display.max_colwidth', -1)
# print(news.head(1))
# print(news[['text', 'category']].head(5))

print(news.groupby(by='category').size())

# bar plot
fig, ax = plt.subplots(1, 1, figsize=(35, 7))
sns.countplot(x='category', data=news)
plt.show()

# pie plot
fig, ax = plt.subplots(1, 1, figsize=(15, 15))
news['category'].value_counts().plot.pie(autopct='%1.1f%%')
plt.show()

total_authors = news.authors.nunique()
news_counts = news.shape[0]
print('Total Number of authors : ', total_authors)
print('avg articles written by per author: ' + str(news_counts // total_authors))
print('Total news counts : ' + str(news_counts))

# count the occurrence of each author
authors_news_counts = news.authors.value_counts()
# print(authors_news_counts)

sum_contribution = 0
author_count = 0

# count only author that has 80 or more articles
for author_contribution in authors_news_counts:
    author_count += 1
    if author_contribution < 80:
        break
    sum_contribution += author_contribution

print('{} of news is contributed by {} authors i.e  {} % of news is contributed by {} % of authors'.
      format(sum_contribution, author_count, format((sum_contribution * 100 / news_counts), '.2f'),
             format((author_count * 100 / total_authors), '.2f')))

print(news.authors.value_counts()[0:10])

'''
If all authors are writing only of few categories of news then we can consider the author feature as well for modeling.
'''
author_name = 'Lee Moran'
# author_name = 'Ed Mazza'
particular_author_news = news[news['authors'] == author_name]
df = particular_author_news.groupby(by='category')['text'].count()
print(df)

fig, ax = plt.subplots(1, 1, figsize=(20, 20))
df.plot.pie(autopct='%1.1f%%')
plt.show()

'''
We can observe that even though authors are writing for almost all category but majority of their contribution
'''
