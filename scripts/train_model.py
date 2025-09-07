from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

def train_models(data):
    x_train, x_test, y_train, y_test = train_test_split(
        data['text'], data['class'], test_size=0.25
    )

    vectorizer = TfidfVectorizer()
    x_train = vectorizer.fit_transform(x_train)
    x_test = vectorizer.transform(x_test)

    log_reg = LogisticRegression()
    log_reg.fit(x_train, y_train)

    dtree = DecisionTreeClassifier()
    dtree.fit(x_train, y_train)

    return log_reg, dtree, x_train, x_test, y_train, y_test
