from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score


iris = datasets.load_iris() 
X,y = iris.data,iris.target

random_space = np.arange(1, 11)
test_accuracys = np.empty(len(random_space))
train_accuracys = np.empty(len(random_space))
train_accuracys = []
test_accuracys = []
for i, k in enumerate(random_space):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, 
                                                        random_state=k, stratify=y)
    dt = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=6)   
    dt.fit(X_train,y_train)
    
    train_accuracy=dt.score(X_train,y_train)
    train_accuracys.append(train_accuracy)
    train_accuracys[i]=train_accuracy
    
    test_accuracy=dt.score(X_test,y_test)
    test_accuracys.append(test_accuracy)
    test_accuracys[i]=test_accuracy
    print('random state: %2d, Score for testing sample: %.3f, Score for training sample: %.3f' % (k, test_accuracy,train_accuracy))
print('Mean of scores for training sample: %.3f'% np.mean(train_accuracys))
print('Standard deviation of scores for training sample: %.3f'% np.std(train_accuracys))

print('Mean of scores for testing sample: %.3f'% np.mean(test_accuracys))
print('Standard deviation of scores for testing sample: %.3f '% np.std(test_accuracys))



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                        random_state=2, stratify=y)
scores = cross_val_score(estimator=dt, X=X_train, y=y_train, cv=10, n_jobs=1)
print('CV accuracy scores of training samples: %s' % scores)
print('mean of scores of training samples: %.3f'% np.mean(scores))
print('mean of scores of training samples: %.3f'% np.std(scores))

scores = cross_val_score(estimator=dt, X=X_test, y=y_test, cv=10, n_jobs=1)
print('CV accuracy scores of testing samples: %s' % scores)
print('mean of scores of testing samples: %.3f'% np.mean(scores))
print('mean of scores of testing samples: %.3f'% np.std(scores))


print("My name is Xiaoyu Yuan")
print("My NetID is: 664377413")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
