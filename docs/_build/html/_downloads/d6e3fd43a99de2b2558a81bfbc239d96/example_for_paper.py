'''A simple example for the paper'''

from MRCpy import MRC
from MRCpy.datasets import load_mammographic
from sklearn.model_selection import train_test_split

# Load the mammographic dataset
X, Y = load_mammographic(return_X_y=True)

# Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create the MRC classifier using default loss (0-1)
clf = MRC()

# Fit the classifier on the training data
clf.fit(X_train, y_train)

# Bounds on the classification error (only for MRC)
lower_error = clf.get_lower_bound()
upper_error = clf.upper_

# Compute the accuracy on the test set
accuracy = clf.score(X_test, y_test)

