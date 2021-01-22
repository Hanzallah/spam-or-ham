'''
# Homework 01 - Spam SMS Detection
# Hanzallah Azim Burney
'''
import os
import sys
import numpy as np

def create_vocabulary(tokenized_sms):
  vocabulary = np.array([])
  # read file and add every sms to a numpy array
  with open(tokenized_sms, 'r') as file:
    for line in file:
      cur_line = np.array(line.replace('\n','').split(','))
      vocabulary = np.append(vocabulary, cur_line) 

  # obtain all unique words in dataset (vocabulary)
  vocabulary, index = np.unique(vocabulary, return_index=True)
  vocabulary = vocabulary[index.argsort()]
  return vocabulary

"""Feature Matrix"""
def create_feature_matrix(tokenized_sms, vocabulary):
  feature_matrix = []

  # read file and create a feature matrix
  with open(tokenized_sms, 'r') as file:
    for line in file:
      fill_vocab = np.array([0]*len(vocabulary))
      cur_line = np.array(line.replace('\n','').split(','))
      cur_vocab, cur_frequency = np.unique(cur_line, return_counts=True) 
      cur_dict = dict(zip(cur_vocab, cur_frequency))
      for key in cur_dict.keys():
        fill_vocab[np.where(vocabulary == key)] += cur_dict[key]
      feature_matrix.append(fill_vocab)

  feature_matrix = np.asarray(feature_matrix)
  return feature_matrix

def create_sets(feature_matrix, labels_file):
  labels = np.array([])
  with open(labels_file, 'r') as file:
    for line in file:
      labels = np.append(labels, int(line[0]))

  # split feature matrix into training and testing sets
  train_set = feature_matrix[:4460]
  test_set = feature_matrix[:1112]
  train_labels = labels[:4460]
  test_labels = labels[:1112]

  return (train_set, test_set, train_labels, test_labels)

"""Multinomial Naive Bayes Model"""

# calculate the MLE estimators
def mle_estimate(train_set, train_labels):
  y_spam = []
  y_ham = []
  T_spam = np.array([])
  T_ham = np.array([])
  theta_spam = np.array([])
  theta_ham = np.array([])

  # get the spam and ham messages into their own array from train set
  y_ham = [train_set[i,:] for i in np.where(train_labels == 0)][0]
  y_spam = [train_set[i,:] for i in np.where(train_labels == 1)][0]

  y_ham = np.asarray(y_ham)
  y_spam = np.asarray(y_spam)

  # number of occurrences of the word j in spam SMSs in the training set
  T_spam = np.sum(y_spam, axis=0)
  T_ham = np.sum(y_ham, axis=0)

  # estimate the probability that a particular word in a spam and ham SMS will be the j-th word of the vocabulary
  theta_spam = np.divide(T_spam, np.sum(T_spam))
  theta_ham = np.divide(T_ham, np.sum(T_ham))

  # estimates the probability that any particular SMS will be spam
  spam_prob = len(y_spam) / len(train_set)
  
  return (theta_spam, theta_ham, spam_prob)

# multinomial naive bayes
def naive_bayes(spam_prob, theta_spam, theta_ham, test_set, test_labels):
  actual = test_labels
  predictions = []

  # naive bayes model
  # probability that ith SMS is spam
  y_pspam = np.log(spam_prob) + np.sum((test_set * np.log(theta_spam)),axis=1)
  y_pham = np.log(1-spam_prob) + np.sum((test_set * np.log(theta_ham)),axis=1)

  # add prediction
  predictions = [1 if y_pspam[i] >= y_pham[i] else 0 for i in range(len(test_set))]

  # accuracy metric calculation
  TN = 0
  FN = 0
  TP = 0
  FP = 0

  for a_y, p_y in zip(actual, predictions):
    if a_y == 0 and p_y == 0:
      TN += 1
    elif a_y == 0 and p_y == 1:
      FP += 1
    elif a_y == 1 and p_y == 0:
      FN += 1
    elif a_y == 1 and p_y == 1:
      TP += 1
  
  # print("TN:", str(TN))
  # print("FP:", str(FP))
  # print("FN:", str(FN))
  # print("TP:", str(TP))
  
  return np.around(((TN+TP)/len(actual))*100, decimals=2)

# MAP estimate of theta using a Dirichlet prior
def map_estimate(train_set, train_labels, alpha=1):
  alpha = alpha
  y_spam = []
  y_ham = []
  T_spam = np.array([])
  T_ham = np.array([])
  theta_spam = np.array([])
  theta_ham = np.array([])

  # get the spam and ham messages into their own array from train set
  y_ham = [train_set[i,:] for i in np.where(train_labels == 0)][0]
  y_spam = [train_set[i,:] for i in np.where(train_labels == 1)][0]

  y_ham = np.asarray(y_ham)
  y_spam = np.asarray(y_spam)

  # number of occurrences of the word j in spam/ham SMSs in the training set
  # add alpha value to each element in T_spam/T_ham array
  T_spam = np.sum(y_spam, axis=0)
  T_spam = T_spam + alpha

  T_ham = np.sum(y_ham, axis=0)
  T_ham = T_ham + alpha

  # estimate the probability that a particular word in a spam and ham SMS will be the j-th word of the vocabulary
  # add alpha * V to the denominator in theta_spam/theta_ham
  theta_spam = np.divide(T_spam, (np.sum(T_spam) + alpha * len(T_spam)))
  theta_ham = np.divide(T_ham, (np.sum(T_ham) + alpha * len(T_ham)))

  # estimates the probability that any particular SMS will be spam
  spam_prob = len(y_spam) / len(train_set)
  ham_prob = 1 - spam_prob

  return (theta_spam, theta_ham, spam_prob)

"""Feature Selection"""

# forward selection
def forward_selection(feature_matrix_r,  train_labels, test_labels):
  selected_features_indices = []
  prev_acc = -1
  curr_acc = 0
  G = []
  scores = []

  while curr_acc - prev_acc > 0.01:
    selected_features_indices = G[scores.index(max(scores))] if len(scores) > 0 else selected_features_indices

    prev_acc = curr_acc 
    curr_selected = []
    G = []
    scores = []

    for i in range(len(feature_matrix_r[0])):
      if i not in selected_features_indices:
        G.append(selected_features_indices + [i])

        # get the modified set based on the indices of G      
        modified_set = feature_matrix_r[:, G[-1]]
        
        # redefine train and test sets
        train_set = modified_set[:4460]
        test_set = modified_set[:1112]

        # redefine parameters with dirichlet prior and test
        (re_theta_spam, re_theta_ham, spam_prob) = map_estimate(train_set, train_labels, alpha=1)
        scores.append(naive_bayes(spam_prob, re_theta_spam, re_theta_ham, test_set, test_labels))

    curr_acc = max(scores) if len(scores) > 0 else curr_acc
  return selected_features_indices



# feature selection using the frequency of words
def frequency_selection(feature_matrix_r, train_labels, test_labels):
  feature_matrix_r_collapse = np.sum(feature_matrix_r, axis = 0)
  desc_feature_tuples = {i:feature_matrix_r_collapse[i] for i in range(len(feature_matrix_r_collapse))}
  desc_feature_tuples = sorted(desc_feature_tuples.items(), key=lambda x: x[1], reverse=True)

  frequency_accuracy = []

  for i in range(1, len(desc_feature_tuples)):
    score = 0
    G = [desc_feature_tuples[:i][j][0] for j in range(i)]

    # get the modified set based on the indices of G      
    modified_set = feature_matrix_r[:, G]
        
    # redefine train and test sets
    train_set = modified_set[:4460]
    test_set = modified_set[:1112]
 
    # redefine parameters with dirichlet prior and test
    (re_theta_spam, re_theta_ham, spam_prob) = map_estimate(train_set, train_labels, alpha=1)
    score = naive_bayes(spam_prob, re_theta_spam, re_theta_ham, test_set, test_labels)

    frequency_accuracy.append((i, score))

  return frequency_accuracy

def main():
  # get the tokenized_corpus file
  data_root = './'
  tokenized_sms = os.path.join(data_root, 'tokenized_corpus.csv')
  vocabulary = create_vocabulary(tokenized_sms)
  feature_matrix = create_feature_matrix(tokenized_sms, vocabulary)

  # save feature matrix as csv
  np.savetxt(fname="feature_set.csv",X = feature_matrix, fmt='%.2f', delimiter=',')

  # get the labels
  labels_file = os.path.join(data_root, 'labels.csv')
  (train_set, test_set, train_labels, test_labels) = create_sets(feature_matrix, labels_file)

  # get accuracy with mle estimate
  (theta_spam, theta_ham, spam_prob) = mle_estimate(train_set, train_labels)
  accuracy = naive_bayes(spam_prob, theta_spam, theta_ham, test_set, test_labels)
  print("The accuracy obtained is " + str(accuracy) + "%")
  # save accuracy to file
  np.savetxt(fname="test_accuracy.csv", X = [accuracy],  fmt='%.2f')

  # get accuracy with map estimate
  (map_spam, map_ham, map_spam_prob) = map_estimate(train_set, train_labels, alpha=1)
  accuracy = naive_bayes(map_spam_prob, map_spam, map_ham, test_set, test_labels)
  print("The accuracy obtained is " + str(accuracy) + "%")
  # save accuracy to file
  np.savetxt(fname="test_accuracy_laplace.csv", X = [accuracy],  fmt='%.2f')

  # create new reduced feature set of words that occur atleast 10 times
  feature_matrix_collapse = np.sum(feature_matrix, axis = 0)
  feature_matrix_r = np.array([])
  feature_matrix_r = [feature_matrix[:, i] for i in np.where(feature_matrix_collapse >= 10)][0]

  selected_features_indices = forward_selection(feature_matrix_r, train_labels, test_labels)
  print(len(selected_features_indices),"features selected using forward selection")
  # save feature indices to file
  np.savetxt(fname="forward_selection.csv", X = selected_features_indices, fmt='%.2f',  delimiter="\n")

  frequency_accuracy = frequency_selection(feature_matrix_r, train_labels, test_labels)
  # save feature indices to file
  np.savetxt(fname="frequency_selection.csv", X = frequency_accuracy, fmt='%.2f', delimiter=',', newline='\n')
    

if __name__ == "__main__":
    main()