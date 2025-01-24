'''
Start code for Project 1-Part 2: Classification
CSE583/EE552 PRML
TA: Keaton Kraiger, 2023
TA: Shimian Zhang, 2023

Your Details: (The below details should be included in every python
file that you add code to.)
{
}
'''

import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from deep_learning_classifier import MLP
from sklearn.metrics import accuracy_score

def deep_learning(train_feats_proj,train_labels,test_feats_proj,test_labels,input_dim=2,output_dim=10,hidden_dim=64,num_layers=3,batch_size=64,learning_rate=0.001,epochs=100):

    model = MLP(input_dim, output_dim, hidden_dim, nn.ReLU, num_layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    from torch.optim.lr_scheduler import StepLR

    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    train_feats_proj = torch.tensor(train_feats_proj, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.long)

    train_dataset = TensorDataset(train_feats_proj, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_feats_proj = torch.tensor(test_feats_proj, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.long)

    test_dataset = TensorDataset(test_feats_proj, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_data, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_data)  # Model call
            loss = criterion(outputs, batch_labels)
            loss.backward()

            optimizer.step()  # Update model parameters
            total_loss += loss.item()
        scheduler.step()  # Update learning rate if using a scheduler (optional)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            outputs = model(batch_data)
            _, predicted = torch.max(outputs, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

    accuracy = correct / total * 100
    print(f"Deep Learning Test Accuracy: {accuracy:.2f}%")

    correct = 0
    total = 0
    with torch.no_grad():
        for batch_data, batch_labels in train_loader:
            outputs = model(batch_data)
            _, predicted = torch.max(outputs, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

    accuracy = correct / total * 100
    print(f"Deep Learning Train Accuracy: {accuracy:.2f}%")



def perform_traditional(train_feats_proj,train_labels,test_feats_proj,test_labels):
    # Train the classifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import classification_report

    classifiers = {
        "Decision Tree 3": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Decision Tree 5": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Decision Tree 7": DecisionTreeClassifier(max_depth=7, random_state=42),
        "Decision Tree 10": DecisionTreeClassifier(max_depth=10, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42),
        "K-Nearest Neighbors (KNN) 5": KNeighborsClassifier(n_neighbors=5),
        "K-Nearest Neighbors (KNN) 7": KNeighborsClassifier(n_neighbors=5),
        "K-Nearest Neighbors (KNN) 9": KNeighborsClassifier(n_neighbors=5),
        "K-Nearest Neighbors (KNN) 4": KNeighborsClassifier(n_neighbors=5),
        "LDA": LinearDiscriminantAnalysis(),
        "deep_learning": deep_learning,
    }

    for name, clf in classifiers.items():
        if name != "deep_learning":
            print(f"\n--- {name} Classifier ---")

            # Train the classifier
            clf.fit(train_feats_proj, train_labels)

            # Predict the labels of the training and testing data
            pred_train_labels = clf.predict(train_feats_proj)
            pred_test_labels = clf.predict(test_feats_proj)

            # Print classification reports
            # print("\nTraining Data Classification Report:")
            # print(classification_report(train_labels, pred_train_labels))
            accuracy = accuracy_score(train_labels, pred_train_labels)
            print(f"Training Accuracy: {accuracy * 100:.2f}%")

            # print("\nTest Data Classification Report:")
            # print(classification_report(test_labels, pred_test_labels))
            accuracy = accuracy_score(test_labels, pred_test_labels)
            print(f"Testing Accuracy: {accuracy * 100:.2f}%")

        else:
            clf(train_feats_proj,train_labels,test_feats_proj,test_labels)


def viz_desc_bounds(classifier, feats, labels, idxA, idxB):
    """
    Visualizes the decision boundaries of a classifier trained on two features of the dataset.
    Args:
        classifier: linear classifier trained on 2 features.
        feats: features to be used for visualization.
        labels: labels to be used for visualization.
        idxA & idxB: indices of the features to be used for visualization.
    """
    if not os.path.exists('results'):
        os.makedirs('results')

    ys = np.sort(np.unique(labels))
    y_ind = np.searchsorted(ys, labels)

    fig, ax = plt.subplots()

    x0, x1 = feats[:, 0], feats[:, 1]
    all_feats = np.concatenate((x0, x1))
    pad = np.percentile(all_feats, 60)

    x_min, x_max = x0.min() - pad, x0.max() + pad
    y_min, y_max = x1.min() - pad, x1.max() + pad
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    preds = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    preds = preds.reshape(xx.shape)

    lut = np.sort(np.unique(labels))
    ind = np.searchsorted(lut,preds)

    markers = ["o", "v", "P", "X", "s", "p", "h", "d"]
    ax.contourf(xx, yy, preds, cmap=plt.cm.Pastel1, alpha=0.8)
    for i in range(len(lut)):
        ax.scatter(x0[y_ind == i], x1[y_ind == i], color=plt.cm.jet(i/len(lut)), s=50, edgecolors='k', marker=markers[i])

    ax.set_xlabel(f'Feature {idxA}')
    ax.set_ylabel(f'Feature {idxB}')
    ax.set_title('Decision Boundary')

    handles = []
    markers = ["o", "v", "P", "X", "s", "p", "h", "d"]
    handles = [plt.plot([],[],color=plt.cm.jet(i/len(lut)), ls="", marker=markers[i])[0] for i in range(len(lut))]
    labels = [f'Class {i}' for i in lut]
    ax.legend(handles, labels, loc='upper right')
    plt.show()
    plt.savefig('results/decision_boundary.png')
    plt.close(fig)

def load_dataset(dataset='taiji', verbose=True, subject_index=3):
    '''
    Loads the taiji dataset.
    Args:
        dataset: name of the dataset to load. Currently only taiji is supported.
        verbose: print dataset information if True.
        subject_index: subject index to use for LOSO. The subject with this index will be used for testing.

    Returns (all numpy arrays):
        train_feats: training features.
        train_labels: training labels.
        test_feats: testing features.
        test_labels: testing labels.
    '''

    if dataset == 'taiji':
        labels = np.loadtxt("data/taiji/taiji_labels.csv", delimiter=",", dtype=int)
        person_idxs = np.loadtxt("data/taiji/taiji_person_idx.csv", delimiter=",", dtype=int)
        feats = np.loadtxt("data/taiji/taiji_quat.csv", delimiter=",", dtype=float)
        # Combine repeated positions
        labels[labels == 4] = 2
        labels[labels == 8] = 6

        # Remove static dimensions. Get mask of all features with zero variance
        feature_mask = np.var(feats, axis=1) > 0

        # Train mask
        train_mask = person_idxs != subject_index

        train_feats = feats[feature_mask, :][:, train_mask].T
        train_labels = labels[train_mask].astype(int)
        test_feats = feats[feature_mask, :][:, ~train_mask].T
        test_labels = labels[~train_mask].astype(int)


    if verbose:
        print(f'{dataset} Dataset Loaded')
        print(f'\t# of Classes: {len(np.unique(train_labels))}')
        print(f'\t# of Features: {train_feats.shape[1]}')
        print(f'\t# of Training Samples: {train_feats.shape[0]}')
        print('\t# per Class in Train Dataset:')
        for cls in np.unique(train_labels):
            print (f'\t\tClass {cls}: {np.sum(train_labels == cls)}')
        print(f'\t# of Testing Samples: {test_feats.shape[0]}')
        print('\t# per Class in Test Dataset:')
        for clas in np.unique(test_labels):
            print(f'\t\tClass {clas}: {np.sum(test_labels == clas)}')

    return train_feats, train_labels, test_feats, test_labels

def plot_conf_mats(dataset, **kwargs):
    """
    Plots the confusion matrices for the training and testing data.
    Args:
        dataset: name of the dataset.
        train_labels: training labels.
        pred_train_labels: predicted training labels.
        test_labels: testing labels.
        pred_test_labels: predicted testing labels.
    """

    train_labels = kwargs['train_labels']
    pred_train_labels = kwargs['pred_train_labels']
    test_labels = kwargs['test_labels']
    pred_test_labels = kwargs['pred_test_labels']

    train_confusion = confusion_matrix(train_labels, pred_train_labels)
    test_confusion = confusion_matrix(test_labels, pred_test_labels)

    # Plot the confusion matrices as seperate figures
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=train_confusion, display_labels=np.unique(train_labels))
    disp.plot(ax=ax, xticks_rotation='vertical')
    ax.set_title('Training Confusion Matrix')
    plt.tight_layout()
    plt.savefig('results/train_confusion.png', bbox_inches='tight', pad_inches=0)

    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=test_confusion, display_labels=np.unique(test_labels))
    disp.plot(ax=ax, xticks_rotation='vertical')
    ax.set_title('Testing Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'results/{dataset}_test_confusion.png', bbox_inches='tight', pad_inches=0)

def example_decision_boundary(dataset='taiji', indices=[0, 6]):
    """
    An example of how to visualize the decision boundary of a classifier.
    """
    train_feats, train_labels, test_feats, test_labels = load_dataset(dataset=dataset)

    dc_train_feats = train_feats[:, indices]
    dc_test_feats = test_feats[:, indices]

    # Example Linear Discriminant Analysis classifier with sklearn's implemenetation
    clf = LinearDiscriminantAnalysis()
    clf.fit(dc_train_feats, train_labels)

    # Visualize the decision boundary
    viz_desc_bounds(clf, dc_test_feats, test_labels, indices[0], indices[1])

def example_classification(dataset='taiji'):
    """
    An example of performing classification. Except you will need to first project the data.
    """
    train_feats, train_labels, test_feats, test_labels = load_dataset(dataset=dataset)

    # Example Linear Discriminant Analysis classifier with sklearn's implemenetation
    clf = LinearDiscriminantAnalysis()
    clf.fit(train_feats, train_labels)

    # Predict the labels of the training and testing data
    pred_train_labels = clf.predict(train_feats)
    pred_test_labels = clf.predict(test_feats)

    # Get statistics
    plot_conf_mats(dataset, train_labels=train_labels, pred_train_labels=pred_train_labels, test_labels=test_labels, pred_test_labels=pred_test_labels)

# TODO: Implement your Fisher projection
def fisher_projection(train_feats, train_labels):
    mean_total = np.mean(train_feats, axis=0)
    means = {}
    # calculating mean of each class (m_k)
    for k in np.unique(train_labels):
        means[k] = np.mean(train_feats[train_labels == k], axis=0)

    # Calculating variance within classes
    S_w = np.zeros((train_feats.shape[1], train_feats.shape[1]))
    for k, mean_k in means.items():
        # S_w += (x_k-m_k)T . (x_k-m_k)
        S_w += (train_feats[train_labels == k] - mean_k).T.dot(train_feats[train_labels == k] - mean_k)

    # Calculating variance between classes
    S_b = np.zeros((train_feats.shape[1], train_feats.shape[1]))
    for k, mean_k in means.items():
        n_k = train_feats[train_labels == k].shape[0]
        delta_mean = (mean_k - mean_total).reshape(train_feats.shape[1], 1)
        # S_b += n_k (m_k - mean).(m_k - mean)T
        S_b += n_k * (delta_mean).dot(delta_mean.T)

    # J(W) = S_w^-1 * S_b
    W = np.linalg.inv(S_w).dot(S_b)

    # Solve for eigen values and eigen vectors of Sw^-1 * Sb
    eigen_values, eigen_vectors = np.linalg.eigh(W)

    eigen_vectors = eigen_vectors.T
    idx = np.argsort(abs(eigen_values))[::-1]
    eigen_vectors = eigen_vectors[idx]
    return eigen_vectors[0: 2]

# TODO: Use the exisintg functions load_dataset and plot_conf_mats. Using a classifier (either one you write
# or an imported sklearn function), perform classification on the fisher projected data.
def classification(dataset='taiji'):
    train_feats, train_labels, test_feats, test_labels = load_dataset()
    train_eigens = fisher_projection(train_feats, train_labels)
    # print(train_eigens)
    # project the features to lower dimensions
    train_feats_proj = np.dot(train_feats,  train_eigens.T)
    test_feats_proj = np.dot(test_feats,  train_eigens.T)
    print("Shape",train_feats_proj.shape,test_feats_proj.shape)
    perform_traditional(train_feats_proj,train_labels,test_feats_proj,test_labels)




def main():
    # example_decision_boundary()
    classification()


if __name__ == '__main__':
    main()

