import argparse
import random
from sklearn.datasets import fetch_20newsgroups
from sklearn.base import is_classifier
import numpy as np
random.seed(42)
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier

###### PART 1
#DONT CHANGE THIS FUNCTION
def part1(samples):
    #extract matrix
    X = extract_matrix(samples)
    assert type(X) == np.ndarray
    print("Example sample feature vec: ", X[0])
    print("Data shape: ", X.shape)
    return X

    
  
   
def extract_matrix(samples):
   print("Extracting matrix ...")
   starting_index_of_word = 0
   all_dicts_of_rows = []

   map_word_to_index = {}
   for sample in samples:
       #print(sample)
       count_of_rows = {}
       words = [w.lower() for w in sample.split() if w.isalpha()]
       for w in words:
           if w not in map_word_to_index:                   
               map_word_to_index[w] = starting_index_of_word
               inner_index = starting_index_of_word
               starting_index_of_word += 1            
           else:
               inner_index = map_word_to_index[w]               
           if inner_index not in count_of_rows:
               count_of_rows[inner_index] = 1
           else:
               count_of_rows[inner_index] += 1
       all_dicts_of_rows.append(count_of_rows)
   num_of_rows = len(samples)
   num_of_columns = len(map_word_to_index)
   #print(all_dicts_of_rows) 
   matrix = np.zeros((num_of_rows, num_of_columns))
   for number, dict in enumerate(all_dicts_of_rows):
       for key, value in dict.items():
           matrix[number, key] = value
   sum_of_columns = np.sum(matrix, axis=0)
   print(sum_of_columns)
   filtered_words = []
   for j in range(len(sum_of_columns)):
       if sum_of_columns[j] < 15:
           filtered_words.append(j)
   filtered_matrix = np.delete(matrix, filtered_words, axis=1)
   #print(filtered_matrix)
   return filtered_matrix
    




##### PART 2
#DONT CHANGE THIS FUNCTION
def part2(X, n_dim):
    #Reduce Dimension
    print("Reducing dimensions ... ")
    X_dr = reduce_dim(X, n=n_dim)
    assert X_dr.shape != X.shape
    assert X_dr.shape[1] == n_dim
    print("Example sample dim. reduced feature vec: ", X[0])
    print("Dim reduced data shape: ", X_dr.shape)
    return X_dr


def reduce_dim(X,n=10):
    pca = PCA(n_components=n)
    dim_red = pca.fit_transform(X)
    return dim_red





##### PART 3
#DONT CHANGE THIS FUNCTION EXCEPT WHERE INSTRUCTED
def get_classifier(clf_id):
    if clf_id == 1:
        clf = GaussianNB() # <--- REPLACE THIS WITH A SKLEARN MODEL
    elif clf_id == 2:
        clf = DecisionTreeClassifier()  # <--- REPLACE THIS WITH A SKLEARN MODEL
    else:
        raise KeyError("No clf with id {}".format(clf_id))

    assert is_classifier(clf)
    print("Getting clf {} ...".format(clf.__class__.__name__))
    return clf

#DONT CHANGE THIS FUNCTION
def part3(X, y, clf_id):
    #PART 3
    X_train, X_test, y_train, y_test = shuffle_split(X,y)

    #get the model
    clf = get_classifier(clf_id)

    #printing some stats
    print()
    print("Train example: ", X_train[0])
    print("Test example: ", X_test[0])
    print("Train label example: ",y_train[0])
    print("Test label example: ",y_test[0])
    print()


    #train model
    print("Training classifier ...")
    train_classifer(clf, X_train, y_train)


    # evalute model
    print("Evaluating classcifier ...")
    evalute_classifier(clf, X_test, y_test)


def shuffle_split(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=8)
    return X_train, X_test, y_train, y_test

def train_classifer(clf, X, y):
    assert is_classifier(clf)
    clf.fit(X,y)
    return clf


def evalute_classifier(clf, X, y):
    assert is_classifier(clf)
    clf_accuracy = accuracy_score(y, clf.predict(X))    
    clf_precision = precision_score(y, clf.predict(X), average='weighted')
    clf_recall = recall_score(y, clf.predict(X), average='weighted')
    clf_f1 = f1_score(y, clf.predict(X), average='weighted')
    print('accuracy score: ', clf_accuracy, ', precision score: ', clf_precision, ', recall score: ', clf_recall, ', F1 score: ', clf_f1)
    

    


######
#DONT CHANGE THIS FUNCTION
def load_data():
    print("------------Loading Data-----------")
    data = fetch_20newsgroups(subset='all', shuffle=True, random_state=42)
    print("Example data sample:\n\n", data.data[0])
    print("Example label id: ", data.target[0])
    print("Example label name: ", data.target_names[data.target[0]])
    print("Number of possible labels: ", len(data.target_names))
    return data.data, data.target, data.target_names


#DONT CHANGE THIS FUNCTION
def main(model_id=None, n_dim=False):

    # load data
    samples, labels, label_names = load_data()


    #PART 1
    print("\n------------PART 1-----------")
    X = part1(samples)

    #part 2
    if n_dim:
        print("\n------------PART 2-----------")
        X = part2(X, n_dim)

    #part 3
    if model_id:
        print("\n------------PART 3-----------")
        part3(X, labels, model_id)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n_dim",
                        "--number_dim_reduce",
                        default=False,
                        type=int,
                        required=False,
                        help="int for number of dimension you want to reduce the matrix for")

    parser.add_argument("-m",
                        "--model_id",
                        default=False,
                        type=int,
                        required=False,
                        help="id of the classifier you want to use")

    args = parser.parse_args()
    main(   
            model_id=args.model_id, 
            n_dim=args.number_dim_reduce
            )