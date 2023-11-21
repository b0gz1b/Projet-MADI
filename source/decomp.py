import numpy as np
from sklearn.tree import DecisionTreeClassifier

def OVA(X,y):
    """
    One vs All
    """
    classifiers = []
    decomps = []
    labels = np.unique(y)
    s_labels = set(labels)
    for i in range(len(labels)):
        y_i = np.where(y==labels[i],1,-1)
        clf = DecisionTreeClassifier()
        clf.fit(X,y_i)
        classifiers.append(clf)
        decomps.append((set(labels[i]),s_labels-set(labels[i])))
    return classifiers, decomps
        

def OVO(X,y):
    """
    One vs One
    """
    classifiers = []
    decomps = []
    labels = np.unique(y)
    for i in range(len(labels)):
        for j in range(i+1,len(labels)):
            X_ij = np.vstack((X[y==labels[i]],X[y==labels[j]]))
            y_ij = np.hstack((np.ones(X[y==labels[i]].shape[0]),-np.ones(X[y==labels[j]].shape[0])))
            clf = DecisionTreeClassifier()
            clf.fit(X_ij,y_ij)
            classifiers.append(clf)
            decomps.append((set(labels[i]),set(labels[j])))
    return classifiers, decomps

def ECOC_dense(X,y):
    """
    Error Correcting Output Codes dense
    """
    classifiers = []
    decomps = []
    labels = np.unique(y)
    n = int(10*np.log2(len(labels)))
    ecoc_pre = np.array([np.concatenate((np.full(shape=i, fill_value=-1), np.full(shape=len(labels)-i, fill_value=1))) for i in range(1,len(labels))]) # Cardinality +1/-1
    ecoc_matrix = np.hstack([np.random.permutation(ecoc_pre[np.random.randint(len(ecoc_pre))])[:,np.newaxis] for _ in range(n)])
    for code in ecoc_matrix.T:
        # We create the coding dictionary
        trad = dict(zip(labels, code))
        y_code = np.array([trad[y_i] for y_i in y])
        clf = DecisionTreeClassifier()
        clf.fit(X,y_code)
        classifiers.append(clf)
        a = set()
        b = set()
        for label in labels:
            if trad[label] == 1:
                a.add(label)
            elif trad[label] == -1:
                b.add(label)
        decomps.append((a,b))
    return classifiers, decomps

def ECOC_sparse(X,y):
    """
    Error Correcting Output Codes sparse
    """
    labels = np.unique(y)
    n = int(15*np.log2(len(labels)))
    classifiers = []
    decomps = []
    labels = np.unique(y)
    ecoc_matrix = np.hstack([np.random.permutation(np.concatenate(([1,-1], np.random.choice([1,-1,0,0], size=len(labels)-2))))[:,np.newaxis] for _ in range(n)])
    for code in ecoc_matrix.T:
        # We create the coding dictionary
        trad = dict(zip(labels, code))
        X_ab = np.vstack([X[y==label] for label in labels if trad[label] != 0])
        y_code = np.array([trad[y_i] for y_i in y if trad[y_i] != 0])
        clf = DecisionTreeClassifier()
        clf.fit(X_ab,y_code)
        classifiers.append(clf)
        a = set()
        b = set()
        for label in labels:
            if trad[label] == 1:
                a.add(label)
            elif trad[label] == -1:
                b.add(label)
        decomps.append((a,b))
    return classifiers, decomps