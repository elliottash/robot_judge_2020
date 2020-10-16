# load dataset
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(return_X_y=True)

# split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y)

# initialize logistic regression with penalty C = 2
logit = LogisticRegression(C=2, solver='liblinear')

# fit to training dataset
logit.fit(X_train, y_train)

# form test set predictions
y_pred = logit.predict(X_test)

# report test set accuracy
print((y_pred == y_test).mean())

# show confusion matrices for test set
confusion_matrix(  # TODO

# report all the metrics for both label classes
from sklearn.metrics import precision_score, recall_score, balanced_accuracy_score, f1_score, roc_auc_score
# TODO:

# given the balance in the label, what metric should we use?
# TODO:

# make a calibration plot
from seaborn import regplot
regplot(y_test, y_pred, x_bins=20)
