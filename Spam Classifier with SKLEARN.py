import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn import svm
from sklearn.model_selection import GridSearchCV

#load dataset

dataframe = pd.read_csv("C:/Users/Matt/Desktop/spam.csv")
print(dataframe.head())
print(dataframe.shape)

#Split data into test and training

x = dataframe["EmailText"]
y = dataframe["Label"]

x_train, y_train = x[0:4457], y[0:4457]
x_test, y_test = x[4457:], y[4457:]

#Feauture Extraction

cv = CountVectorizer()
features = cv.fit_transform(x_train)

#Build Model

model = svm.SVC()

model.fit(features, y_train)

#Steps 5: Test accuracy

print(model.score(cv.transform(x_test), y_test))