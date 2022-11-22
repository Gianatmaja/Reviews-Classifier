from functools import partial

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from models.perceptron import Perceptron
from models.gd import GD
from readers.reviews_dataset import ReviewsDataset
from utils import compute_average_accuracy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, classification_report

from nltk.corpus import stopwords
StopWords = set(stopwords.words('english'))

# Function to count frequency of items in a list
def Count_vals(List): 
    freq = {} 
    for val in List: 
        if (val in freq): 
            freq[val] = freq[val] + 1
        else: 
            freq[val] = 1
    for ind, val in freq.items(): 
        print ("% d : % d"%(ind, val))



def return_same_val(x):
    return x


no_tokenizer = partial(return_same_val)


def load_dataset(filename):
    dataset = ReviewsDataset()
    dataset.load(filename)
    return dataset


# Part I - Feature Engineering
# In this part we will to try out different feature representations
# by using scikit-learn CountVectorizer and TfIdfTransformer.
#
#
# Input: the train, dev and test instances of class ReviewDataset
# Output: returns 3 things: train_X_features, dev_X_features and test_X_features. These should be the
# feature vectors for the train, dev and test splits.
def compute_features(train, dev, test):

    # Step 1. Create the feature vectorizer. Here is an example for binary word features.
    vectorizer = CountVectorizer(
        tokenizer=no_tokenizer,
        lowercase=False,
        binary=True
    )

    # Step 2. Apply the vectorizer on the training, dev and test set.

    # Step 3. return feature vectors for train, dev and test set
    train_X_features = vectorizer.fit_transform(train.X)
    dev_X_features = vectorizer.transform(dev.X)
    test_X_features = vectorizer.transform(test.X)

    return train_X_features, dev_X_features, test_X_features

def compute_features_B(train, dev, test):
    vectorizer = CountVectorizer(
        tokenizer=no_tokenizer,
        lowercase=False,
        binary=True
    )

    train_X_features = vectorizer.fit_transform(train)
    dev_X_features = vectorizer.transform(dev)
    test_X_features = vectorizer.transform(test)

    return train_X_features, dev_X_features, test_X_features

def compute_features_C1(train, dev, test):
    tf_vectorizer = TfidfVectorizer(tokenizer = no_tokenizer, 
                                    lowercase = False, use_idf=False)
    
    train_X_features = tf_vectorizer.fit_transform(train.X)
    dev_X_features = tf_vectorizer.transform(dev.X)
    test_X_features = tf_vectorizer.transform(test.X)
    
    return train_X_features, dev_X_features, test_X_features

def compute_features_C2(train, dev, test):
    tf_vectorizer = TfidfVectorizer(tokenizer = no_tokenizer, 
                                    lowercase = False, use_idf=False)
    
    train_X_features = tf_vectorizer.fit_transform(train)
    dev_X_features = tf_vectorizer.transform(dev)
    test_X_features = tf_vectorizer.transform(test)
    
    return train_X_features, dev_X_features, test_X_features

def compute_features_D1(train, dev, test):
    tf_vectorizer = TfidfVectorizer(tokenizer = no_tokenizer, 
                                    lowercase = False, use_idf=True)
    
    train_X_features = tf_vectorizer.fit_transform(train.X)
    dev_X_features = tf_vectorizer.transform(dev.X)
    test_X_features = tf_vectorizer.transform(test.X)
    
    return train_X_features, dev_X_features, test_X_features

def compute_features_D2(train, dev, test):
    tf_vectorizer = TfidfVectorizer(tokenizer = no_tokenizer, 
                                    lowercase = False, use_idf=True)
    
    train_X_features = tf_vectorizer.fit_transform(train)
    dev_X_features = tf_vectorizer.transform(dev)
    test_X_features = tf_vectorizer.transform(test)
    
    return train_X_features, dev_X_features, test_X_features

def compute_features_E1(train, dev, test, i):
    tf_vectorizer = TfidfVectorizer(tokenizer = no_tokenizer, 
                                    lowercase = False, use_idf=True,
                                    min_df = i)
    
    train_X_features = tf_vectorizer.fit_transform(train.X)
    dev_X_features = tf_vectorizer.transform(dev.X)
    test_X_features = tf_vectorizer.transform(test.X)
    
    return train_X_features, dev_X_features, test_X_features

def compute_features_E2(train, dev, test, i):
    tf_vectorizer = TfidfVectorizer(tokenizer = no_tokenizer, 
                                    lowercase = False, use_idf=True,
                                    min_df = i)
    
    train_X_features = tf_vectorizer.fit_transform(train)
    dev_X_features = tf_vectorizer.transform(dev)
    test_X_features = tf_vectorizer.transform(test)
    
    return train_X_features, dev_X_features, test_X_features

def compute_features2(train, dev, test, j):

    vectorizer = CountVectorizer(
        tokenizer=no_tokenizer,  
        lowercase=False,  
        binary=True,
        ngram_range = (1, j)
    )

    train_X_features = vectorizer.fit_transform(train.X)
    dev_X_features = vectorizer.transform(dev.X)
    test_X_features = vectorizer.transform(test.X)

    return train_X_features, dev_X_features, test_X_features

def compute_features_C1_2(train, dev, test, j):
    tf_vectorizer = TfidfVectorizer(tokenizer = no_tokenizer, 
                                    lowercase = False, use_idf=False,
                                    ngram_range = (1, j))
    
    train_X_features = tf_vectorizer.fit_transform(train)
    dev_X_features = tf_vectorizer.transform(dev)
    test_X_features = tf_vectorizer.transform(test)
    
    return train_X_features, dev_X_features, test_X_features

def compute_features_E1_2(train, dev, test, j):
    tf_vectorizer = TfidfVectorizer(tokenizer = no_tokenizer, 
                                    lowercase = False, use_idf=True,
                                    ngram_range = (1, j))
    
    train_X_features = tf_vectorizer.fit_transform(train.X)
    dev_X_features = tf_vectorizer.transform(dev.X)
    test_X_features = tf_vectorizer.transform(test.X)
    
    return train_X_features, dev_X_features, test_X_features


def main():
    # load datasets
    train_dataset = load_dataset("data/truecased_reviews_train.jsonl")
    dev_dataset = load_dataset("data/truecased_reviews_dev.jsonl")
    test_dataset = load_dataset("data/truecased_reviews_test.jsonl")

    # Part I: Feature Engineering
    
    ########################## Removed stop words ############################
    train_features_B = []
    for sents in train_dataset.X:
        train_clean = [word for word in sents if not word in StopWords]
        train_features_B.append(train_clean)
    
    dev_features_B = []
    for sents in dev_dataset.X:
        dev_clean = [word for word in sents if not word in StopWords]
        dev_features_B.append(dev_clean)
    
    test_features_B = []
    for sents in test_dataset.X:
        test_clean = [word for word in sents if not word in StopWords]
        test_features_B.append(test_clean)
    
    ###################### Checking class proportions ########################
    print('Part I: Feature Engineering\n')
    Count_vals(train_dataset.y)
    Count_vals(dev_dataset.y)
    
    ########################### Baseline model ###############################
    # Step 1. create feature vectors by calling compute_features() with all three datasets as parameters
    train_vecs, dev_vecs, test_vecs = compute_features(train_dataset, dev_dataset, test_dataset)

    # Step 2. train a Naive Bayes Classifier (scikit MultinomialNB() )
    clf = MultinomialNB().fit(train_vecs, train_dataset.y)

    # Step 3. Check performance
    y_pred = clf.predict(dev_vecs)
    F1_A = f1_score(dev_dataset.y, y_pred)
    acc_A = accuracy_score(dev_dataset.y, y_pred)    
    
    ##################### Model B: Remove stop words #########################    
    train_vecs_B, dev_vecs_B, test_vecs_B = compute_features_B(train_features_B, 
                                                               dev_features_B, test_features_B)
    
    clf_B = MultinomialNB().fit(train_vecs_B, train_dataset.y)
    y_pred_B = clf_B.predict(dev_vecs_B)
    
    F1_B = f1_score(dev_dataset.y, y_pred_B)
    acc_B = accuracy_score(dev_dataset.y, y_pred_B)
    
    ######################## Model C1: TF (no idf) ###########################  
    train_vecs_C1, dev_vecs_C1, test_vecs_C1 = compute_features_C1(train_dataset, 
                                                               dev_dataset, test_dataset)
    
    clf_C1 = MultinomialNB().fit(train_vecs_C1, train_dataset.y)
    y_pred_C1 = clf_C1.predict(dev_vecs_C1)
    
    F1_C1 = f1_score(dev_dataset.y, y_pred_C1)
    acc_C1 = accuracy_score(dev_dataset.y, y_pred_C1)
    
    
    ############## Model C1: TF (no idf) & remove stop words #################
    train_vecs_C2, dev_vecs_C2, test_vecs_C2 = compute_features_C2(train_features_B, 
                                                               dev_features_B, test_features_B)
    
    clf_C2 = MultinomialNB().fit(train_vecs_C2, train_dataset.y)
    y_pred_C2 = clf_C2.predict(dev_vecs_C2)
    
    F1_C2 = f1_score(dev_dataset.y, y_pred_C2)
    acc_C2 = accuracy_score(dev_dataset.y, y_pred_C2)
    
    ########################### Model D1: TFIDF ##############################  
    train_vecs_D1, dev_vecs_D1, test_vecs_D1 = compute_features_D1(train_dataset, 
                                                               dev_dataset, test_dataset)
    
    clf_D1 = MultinomialNB().fit(train_vecs_D1, train_dataset.y)
    y_pred_D1 = clf_D1.predict(dev_vecs_D1)
    
    F1_D1 = f1_score(dev_dataset.y, y_pred_D1)
    acc_D1 = accuracy_score(dev_dataset.y, y_pred_D1)
    
    ################# Model D2: TFIDF & remove stop words ####################
    train_vecs_D2, dev_vecs_D2, test_vecs_D2 = compute_features_D2(train_features_B, 
                                                               dev_features_B, test_features_B)
    
    clf_D2 = MultinomialNB().fit(train_vecs_D2, train_dataset.y)
    y_pred_D2 = clf_D2.predict(dev_vecs_D2)
    
    F1_D2 = f1_score(dev_dataset.y, y_pred_D2)
    acc_D2 = accuracy_score(dev_dataset.y, y_pred_D2)    
    
    ##################### Model E1: TFIDF with threshold #####################
    F1_E1 = []
    Acc_E1 = []
    Vocab_size_E = []
    for i in range(2,5):
        train_vecs_E1, dev_vecs_E1, test_vecs_E1 = compute_features_E1(train_dataset, 
                                                                       dev_dataset, 
                                                                       test_dataset, i)
        
        Shape_E = train_vecs_E1.shape[1]
        Vocab_size_E.append(Shape_E)
    
        clf_E1 = MultinomialNB().fit(train_vecs_E1, train_dataset.y)
        y_pred_E1 = clf_E1.predict(dev_vecs_E1)
    
        F1_E1_i = f1_score(dev_dataset.y, y_pred_E1)
        acc_E1_i = accuracy_score(dev_dataset.y, y_pred_E1)
        
        F1_E1.append(F1_E1_i)
        Acc_E1.append(acc_E1_i)
        
    print('Vocab size with threshold 2, 3, and 4, respectively: ', Vocab_size_E)
    print('\n')
    
    ########### Model E1: TFIDF with threshold and stop words removed#########
    F1_E2 = []
    Acc_E2 = []
    for i in range(2,5):
        train_vecs_E2, dev_vecs_E2, test_vecs_E2 = compute_features_E2(train_features_B, 
                                                                       dev_features_B,
                                                                       test_features_B, i)
    
        clf_E2 = MultinomialNB().fit(train_vecs_E2, train_dataset.y)
        y_pred_E2 = clf_E2.predict(dev_vecs_E2)
    
        F1_E2_i = f1_score(dev_dataset.y, y_pred_E2)
        acc_E2_i = accuracy_score(dev_dataset.y, y_pred_E2)
        
        F1_E2.append(F1_E2_i)
        Acc_E2.append(acc_E2_i)
    
    #################### Baseline with ngrams variation ######################
    
    Acc_ngram_Base = []
    F1_ngram_Base = []
    for j in range(2,5):
        train_vecs2, dev_vecs2, test_vecs2 = compute_features2(train_dataset, dev_dataset, test_dataset, j)

        clf2 = MultinomialNB().fit(train_vecs2, train_dataset.y)

        y_pred2 = clf2.predict(dev_vecs2)
        F1_A2 = f1_score(dev_dataset.y, y_pred2)
        acc_A2 = accuracy_score(dev_dataset.y, y_pred2)
        
        F1_ngram_Base.append(F1_A2)
        Acc_ngram_Base.append(acc_A2)
        
    #################### TF Model with ngrams variation ######################
    
    Acc_ngram_TF = []
    F1_ngram_TF = []
    for j in range(2,5):
        train_vecs_C1_2, dev_vecs_C1_2, test_vecs_C1_2 = compute_features_C1_2(train_features_B, dev_features_B, test_features_B, j)

        clf_C1_2 = MultinomialNB().fit(train_vecs_C1_2, train_dataset.y)

        y_pred_C1_2 = clf_C1_2.predict(dev_vecs_C1_2)
        F1_C1_2 = f1_score(dev_dataset.y, y_pred_C1_2)
        acc_C1_2 = accuracy_score(dev_dataset.y, y_pred_C1_2)
        
        F1_ngram_TF.append(F1_C1_2)
        Acc_ngram_TF.append(acc_C1_2)

    ################### TFIDF Model with ngrams variation ####################
    
    Acc_ngram_TFIDF = []
    F1_ngram_TFIDF = []
    for j in range(2,5):
        train_vecs_E1_2, dev_vecs_E1_2, test_vecs_E1_2 = compute_features_E1_2(train_dataset, dev_dataset, test_dataset, j)

        clf_E1_2 = MultinomialNB().fit(train_vecs_E1_2, train_dataset.y)

        y_pred_E1_2 = clf_E1_2.predict(dev_vecs_E1_2)
        F1_E1_2 = f1_score(dev_dataset.y, y_pred_E1_2)
        acc_E1_2 = accuracy_score(dev_dataset.y, y_pred_E1_2)
        
        F1_ngram_TFIDF.append(F1_E1_2)
        Acc_ngram_TFIDF.append(acc_E1_2)

    ############################ Compiling Results ###########################

    Models = ['Binary Count Vectorizer', 'Binary Count Vectorizer (Stop Words Removed)', 'Binary Count Vectorizer with ngram = (1,2)',
              'Binary Count Vectorizer with ngram = (1,3)', 'Binary Count Vectorizer with ngram = (1,4)', 'TF', 'TF (Stop Words Removed)',
              'TF (Stop Words Removed) with ngram = (1,2)', 'TF (Stop Words Removed) with ngram = (1,3)', 'TF (Stop Words Removed) with ngram = (1,4)', 'TF-IDF', 'TF-IDF (Stop Words Removed)', 'TF-IDF (Thres = 2)', 
              'TF-IDF (Thres = 3)', 'TF-IDF (Thres = 4)', 'TF-IDF (Stop Words Removed, Thres = 2)', 'TF-IDF (Stop Words Removed, Thres = 3)', 
              'TF-IDF (Stop Words Removed, Thres = 4)', 'TF-IDF with ngram = (1,2)', 'TF-IDF with ngram = (1,3)', 'TF-IDF with ngram = (1,4)']
    
    Acc = [acc_A, acc_B, Acc_ngram_Base[0], Acc_ngram_Base[1], Acc_ngram_Base[2], acc_C1, acc_C2, Acc_ngram_TF[0], Acc_ngram_TF[1], Acc_ngram_TF[2], 
           acc_D1, acc_D2, Acc_E1[0], Acc_E1[1], Acc_E1[2], Acc_E2[0], Acc_E2[1], Acc_E2[2], Acc_ngram_TFIDF[0], Acc_ngram_TFIDF[1], Acc_ngram_TFIDF[2]]
    
    F1 = [F1_A, F1_B, F1_ngram_Base[0], F1_ngram_Base[1], F1_ngram_Base[2], F1_C1, F1_C2, F1_ngram_TF[0], F1_ngram_TF[1], F1_ngram_TF[2], 
           F1_D1, F1_D2, F1_E1[0], F1_E1[1], F1_E1[2], F1_E2[0], F1_E2[1], F1_E2[2], F1_ngram_TFIDF[0], F1_ngram_TFIDF[1], F1_ngram_TFIDF[2]]
    
    Data = {'Feature Representation':Models, 'Accuracy': Acc, 'F1-Score':F1}
    Results_df = pd.DataFrame(Data)
    print(Results_df) 
    #CountVectorizer with Binary = True and ngram = (1,2) gives the highest accuracy.
    
    
    print('\n')
    print('Part II: Perceptron\n')
    
    
    # Part II: Perceptron Algorithm
    
    ####################### Perceptron Without Shuffle #######################
    
    #Best feature representation (From Part I)
    train_vecs, dev_vecs, test_vecs = compute_features2(train_dataset,
                                                                   dev_dataset,
                                                                   test_dataset, 2)

    # parameters for the perceptron model
    num_epochs = 30
    num_features = train_vecs.shape[1]
    averaged = False

    # Step 1. Initialise model with hyperparameters
    perceptron = Perceptron(num_epochs, num_features, averaged)

    # Step 2. Train model 
    print("Training perceptron for {} epochs without shuffling".format(num_epochs))
    Tr_acc, Dev_acc = perceptron.train(train_vecs, train_dataset.y, dev_vecs, dev_dataset.y, shuffle = False)
    
    Tr_acc_ptron = [float("%.5f"%item) for item in Tr_acc]
    Dev_acc_ptron = [float("%.5f"%item) for item in Dev_acc]
    
    Best_epochs_I = np.argmax(Dev_acc_ptron)
    
    print('Training set accuracy after 30 epochs: {}'.format(Tr_acc_ptron))
    print('Development set accuracy after 30 epochs: {}'.format(Dev_acc_ptron))
    print('Best number of epochs: {}'.format(Best_epochs_I+1))
    
    # Plotting results
    plt.figure(figsize = (12, 8))
    plt.plot(np.arange(1, (num_epochs+1)), Tr_acc_ptron, c = 'b', label = 'Train data', linestyle = '-', marker = 'o')
    plt.plot(np.arange(1, (num_epochs+1)), Dev_acc_ptron, c = 'r', label = 'Development data', linestyle = '-', marker = 'o')
    plt.title('Training & Development Set Accuracy (Perceptron Without Shuffling)\n')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.legend()
    plt.show()
    
    ######################## Perceptron With Shuffle #########################

    # parameters for the perceptron model
    num_epochs = 30
    num_features = train_vecs.shape[1]
    averaged = False

    # Step 1. Initialise model with hyperparameters
    perceptron = Perceptron(num_epochs, num_features, averaged)

    # Step 2. Train model 
    print("Training perceptron for {} epochs with shuffling".format(num_epochs))
    Tr_acc2, Dev_acc2 = perceptron.train(train_vecs, train_dataset.y, dev_vecs, dev_dataset.y, shuffle = True)
    
    Tr_acc_ptron2 = [float("%.5f"%item) for item in Tr_acc2]
    Dev_acc_ptron2 = [float("%.5f"%item) for item in Dev_acc2]
    
    print('Training set accuracy: {}'.format(Tr_acc_ptron2))
    print('Development set accuracy: {}'.format(Dev_acc_ptron2))
    
    # Plotting results
    plt.figure(figsize = (12, 8))
    plt.plot(np.arange(1, (num_epochs+1)), Tr_acc_ptron, c = 'b', label = 'Train data', linestyle = '-', marker = 'o')
    plt.plot(np.arange(1, (num_epochs+1)), Dev_acc_ptron, c = 'r', label = 'Development data', linestyle = '-', marker = 'o')
    plt.plot(np.arange(1, (num_epochs+1)), Tr_acc_ptron2, c = 'darkorchid', label = 'Train data (Shuffle)', linestyle = '-', marker = 'o')
    plt.plot(np.arange(1, (num_epochs+1)), Dev_acc_ptron2, c = 'darkorange', label = 'Development data (Shuffle)', linestyle = '-', marker = 'o')
    plt.title('Training & Development Set Accuracy (Perceptron With & Without Shuffling)\n')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.legend()
    plt.show()
    
    # Best 
    num_epochs = Best_epochs_I + 1
    num_features = train_vecs.shape[1]
    averaged = False
    
    perceptron = Perceptron(num_epochs, num_features, averaged)
    
    print("Training perceptron for {} epochs without shuffling".format(num_epochs))
    Tr_acc, Dev_acc = perceptron.train(train_vecs, train_dataset.y, dev_vecs, dev_dataset.y, shuffle = False)

    # Step 3. Compute performance on test set
    
    test_preds = perceptron.predict(test_vecs)
    Acc_test = accuracy_score(test_dataset.y, test_preds)
    print('Accuracy on test set: {:.5f}\n'.format(Acc_test))
    
    print('Part III: Gradient Descent\n')
    
    # Part III: Gradient Descent
    
    ############################# 1st Combination ############################
    
    train_vecs, dev_vecs, test_vecs = compute_features2(train_dataset,
                                                        dev_dataset,
                                                        test_dataset, 2)

    # parameters for the gradient descent algorithm
    max_iter = 50
    num_features = train_vecs.shape[1]
    eta = 0.00001
    lam = 0.01

    # Step 1. Initialise model with hyperparameters
    linear_model = GD(max_iter, num_features, eta, lam)

    # Step 2. Train model on a subset of the training set (first 10k examples)
    print("Training model for {} max_iter".format(max_iter))
    
    Loss_LM_Train1, Loss_LM_Dev1, Acc_LM_Train1, Acc_LM_Dev1 = linear_model.train(train_vecs[:10000], train_dataset.y[:10000], dev_vecs, dev_dataset.y)
    Best_iter1 = np.argmin(Loss_LM_Dev1)
    
    print('Average cost for training data: ', Loss_LM_Train1)
    print('Average cost for development data: ', Loss_LM_Dev1)
    print('Accuracy for training data: ', Acc_LM_Train1)
    print('Accuracy for development data: ', Acc_LM_Dev1)
    print('Best number of iterations (Model 1): ', Best_iter1+1)
    print('Accuracy score for best iter (Model 1): ', Acc_LM_Dev1[Best_iter1])
    
    
    ############################# 2nd Combination ############################

    # parameters for the gradient descent algorithm
    max_iter = 50
    num_features = train_vecs.shape[1]
    eta = 0.000015
    lam = 0.01

    # Step 1. Initialise model with hyperparameters
    linear_model = GD(max_iter, num_features, eta, lam)

    # Step 2. Train model on a subset of the training set (first 10k examples)
    print("Training model for {} max_iter".format(max_iter))
    
    Loss_LM_Train2, Loss_LM_Dev2, Acc_LM_Train2, Acc_LM_Dev2 = linear_model.train(train_vecs[:10000], train_dataset.y[:10000], dev_vecs, dev_dataset.y)
    Best_iter2 = np.argmin(Loss_LM_Dev2)
    
    print('Average cost for training data: ', Loss_LM_Train2)
    print('Average cost for development data: ', Loss_LM_Dev2)
    print('Accuracy for training data: ', Acc_LM_Train2)
    print('Accuracy for development data: ', Acc_LM_Dev2)
    print('Best number of iterations (Model 2): ', Best_iter2+1)
    print('Accuracy score for best iter (Model 2): ', Acc_LM_Dev2[Best_iter2])
    
    ############################# 3rd Combination ############################

    # parameters for the gradient descent algorithm
    max_iter = 50
    num_features = train_vecs.shape[1]
    eta = 0.00002
    lam = 0.01

    # Step 1. Initialise model with hyperparameters
    linear_model = GD(max_iter, num_features, eta, lam)

    # Step 2. Train model on a subset of the training set (first 10k examples)
    print("Training model for {} max_iter".format(max_iter))
    
    Loss_LM_Train3, Loss_LM_Dev3, Acc_LM_Train3, Acc_LM_Dev3 = linear_model.train(train_vecs[:10000], train_dataset.y[:10000], dev_vecs, dev_dataset.y)
    Best_iter3 = np.argmin(Loss_LM_Dev3)
    
    print('Average cost for training data: ', Loss_LM_Train3)
    print('Average cost for development data: ', Loss_LM_Dev3)
    print('Accuracy for training data: ', Acc_LM_Train3)
    print('Accuracy for development data: ', Acc_LM_Dev3)
    print('Best number of iterations (Model 3): ', Best_iter3+1)
    print('Accuracy score for best iter (Model 3): ', Acc_LM_Dev3[Best_iter3])
    
    # Plotting Accuracy
    plt.figure(figsize = (12, 8))
    plt.plot(np.arange(1, (max_iter+1)), Acc_LM_Train1, c = 'lightcoral', label = 'Train (1st combination)', linestyle = '-', alpha = 0.7)
    plt.plot(np.arange(1, (max_iter+1)), Acc_LM_Train2, c = 'r', label = 'Train (2nd combination)', linestyle = '-', alpha = 0.7)
    plt.plot(np.arange(1, (max_iter+1)), Acc_LM_Train3, c = 'maroon', label = 'Train (3rd combination)', linestyle = '-', alpha = 0.7)
    plt.plot(np.arange(1, (max_iter+1)), Acc_LM_Dev1, c = 'cyan', label = 'Dev (1st combination)', linestyle = '-', alpha = 0.7)
    plt.plot(np.arange(1, (max_iter+1)), Acc_LM_Dev2, c = 'royalblue', label = 'Dev (2nd combination)', linestyle = '-', alpha = 0.7)
    plt.plot(np.arange(1, (max_iter+1)), Acc_LM_Dev3, c = 'navy', label = 'Dev (3rd combination)', linestyle = '-', alpha = 0.7)
    plt.title('Linear Model Accuracy\n')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.legend()
    plt.show()
    
    # Plotting Avg Loss
    plt.figure(figsize = (12, 8))
    plt.plot(np.arange(1, (max_iter+1)), Loss_LM_Train1, c = 'lightcoral', label = 'Train (1st combination)', linestyle = '-', alpha = 0.7)
    plt.plot(np.arange(1, (max_iter+1)), Loss_LM_Train2, c = 'r', label = 'Train (2nd combination)', linestyle = '-', alpha = 0.7)
    plt.plot(np.arange(1, (max_iter+1)), Loss_LM_Train3, c = 'maroon', label = 'Train (3rd combination)', linestyle = '-', alpha = 0.7)
    plt.plot(np.arange(1, (max_iter+1)), Loss_LM_Dev1, c = 'cyan', label = 'Dev (1st combination)', linestyle = '-', alpha = 0.7)
    plt.plot(np.arange(1, (max_iter+1)), Loss_LM_Dev2, c = 'royalblue', label = 'Dev (2nd combination)', linestyle = '-', alpha = 0.7)
    plt.plot(np.arange(1, (max_iter+1)), Loss_LM_Dev3, c = 'navy', label = 'Dev (3rd combination)', linestyle = '-', alpha = 0.7)
    plt.title('Linear Model Average Loss\n')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Loss')
    plt.ylim(0.75,1.05)
    plt.grid()
    plt.legend()
    plt.show()

    #3rd combination gives the best performance.
    
    # Step 4. Compute performance on test set

    test_preds, loss = linear_model.predict(test_vecs, test_dataset.y) 
    print('Accuracy on test set: {:.5}\n'.format(accuracy_score(test_dataset.y, test_preds)))
    print('Average loss on test set: {:.5}\n'.format(loss))


if __name__ == "__main__":
    main()
