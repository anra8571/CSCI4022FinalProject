# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier.fit
# Description of mathematics behind MLP: https://www.linkedin.com/posts/annarahn_deeplearning-neuralnetworks-math-activity-7171664061187706880-bABO?utm_source=share&utm_medium=member_desktop

from sklearn.neural_network import MLPClassifier # type: ignore
import numpy as np # type: ignore

# Input: all (a Participant object containing the aggregated dataset to cluster using a multi-layer perceptron)
def MLP(all, validation_percent=10):
    # Retain 10% of the data for validation
    X_train = []
    X_label = []
    Y_train = []
    Y_label = []
    for trial in range(0, len(all.fv_reduced)):
        if trial % validation_percent == 0:
            Y_train.append(all.fv_reduced[trial])
            Y_label.append(all.labels[trial])
        else:
            X_train.append(all.fv_reduced[trial])
            X_label.append(all.labels[trial])

    # Convert everything to a np array
    X_train = np.array(X_train)
    X_label = np.array(X_label)
    Y_train = np.array(Y_train)
    Y_label = np.array(Y_label)

    # Create the classifier and train on most of the dataset
    clf = MLPClassifier(max_iter=1000).fit(X_train, X_label)

    # Predict over the set reserved for validation
    mlp_preds = clf.predict(Y_train)

    # Compare the predicted labels to the ground truth labels and report the accuracy
    acc = clf.score(Y_train, Y_label)

    print(f"The MLP accuracy for {all.name} is {acc}")
    return acc