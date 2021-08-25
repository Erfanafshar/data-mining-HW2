from sklearn import neighbors
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing

n_neighbors = 12  # number of neighbors in knn
knn_naive = 1    # knn -> 0, naive -> 1

# train_data
df = pd.read_csv("train.csv")
class_attribute = df[["target"]].copy()
df.drop("target", inplace=True, axis=1)

X = df
normalized_X = preprocessing.robust_scale(X)           # 1.scale 2.minmax_scale 3.robust_scale 4.normalize
y = class_attribute

# test_data
tf = pd.read_csv("test.csv")
class_attribute_test = tf[["target"]].copy()
tf.drop("target", inplace=True, axis=1)


X2 = tf
normalized_X2 = preprocessing.robust_scale(X2)         # 1.scale 2.minmax_scale 3.robust_scale 4.normalize

y2 = class_attribute_test.values.ravel()

predicted = 0
if knn_naive == 0:
    # train KNN
    clf = neighbors.KNeighborsClassifier(n_neighbors)
    clf.fit(normalized_X, y.values.ravel())

    # test KNN
    predicted = clf.predict(normalized_X2)

if knn_naive == 1:
    # train naive
    gnb = GaussianNB()
    clf2 = gnb.fit(normalized_X, y.values.ravel())

    # test naive
    predicted = clf2.predict(normalized_X2)

# assessment
# print(predicted)
accuracy = accuracy_score(y2, predicted)
print(accuracy)
