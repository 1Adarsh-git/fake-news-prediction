import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def plot_class_distribution(data):
    sns.countplot(data=data, x='class', order=data['class'].value_counts().index)
    plt.show()

def plot_top_words(data, n=20):
    vec = CountVectorizer().fit(data['text'])
    bag_of_words = vec.transform(data['text'])
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)[:n]
    df1 = pd.DataFrame(words_freq, columns=['Word', 'Count'])
    df1.set_index('Word')['Count'].plot(kind='bar', figsize=(10, 6), title="Top Words Frequency")
    plt.show()
