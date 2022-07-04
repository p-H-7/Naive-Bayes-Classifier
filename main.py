# CS60050-Machine-Learning
# Assignment - 2: Naive Bayes Classifier
# Group - 1
# Haasita Pinnepu - 19CS30021
# Swapnika Piriya - 19CS30035

# Libraries
import math
import re
import random
from collections import defaultdict
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
#from sklearn.metrics import recall
from sklearn.metrics import confusion_matrix
from math import sqrt


nltk.download('stopwords')


def tokenize(text):
    text = text.lower()
    all_words = re.findall("[a-z0-9]+", text)
    # nltk.download('stopwords')
    filtered_words = [
        word for word in all_words if word not in stopwords.words('english')]
    return (filtered_words)


def createMatrix(df):
    filtered_words = []

    length = df.shape[0]
    for i in range(0, length):
        text = df.loc[i].iat[1]
        k = tokenize(text)
        filtered_words = filtered_words + k

    filtered_words = list(set(filtered_words))

    filtered_words.sort()

    M = defaultdict(lambda: defaultdict(lambda: 0))

    for i in range(0, length):
        text = df.loc[i].iat[1]
        k = set(tokenize(text))
        id = df.loc[i].iat[0]
        for word in k:
            M[id][word] = 1

    return M, length, len(filtered_words), filtered_words


def count_words(training_set, M, filteredWords):
    counts = defaultdict(lambda: {'EAP': 0, 'HPL': 0, 'MWS': 0})
    for item in training_set:
        for word in set(tokenize(item['text'])):
            counts[word][item['author']] += 1
    return counts


def word_probabilities(counts, total_eap, total_hpl, total_mws, k=0.33):
    probs = [{'word': w,
              'eap': (d['EAP'] + k) / (total_eap + 3*k),
              'hpl': (d['HPL'] + k) / (total_hpl + 3*k),
              'mws': (d['MWS'] + k) / (total_mws + 3*k)}
             for w, d in counts.items()]
    return probs


def author_probability(word_probs, text):
    text_words = set(tokenize(text))
    log_prob_if_eap = 0.0
    log_prob_if_hpl = 0.0
    log_prob_if_mws = 0.0
    for d in word_probs:
        if d['word'] in text_words:
            if d['eap'] != 0:
                log_prob_if_eap += math.log(d['eap'])
            if d['hpl'] != 0:
                log_prob_if_hpl += math.log(d['hpl'])
            if d['mws'] != 0:
                log_prob_if_mws += math.log(d['mws'])
        else:
            log_prob_if_eap += math.log(1.0 - d['eap'])
            log_prob_if_hpl += math.log(1.0 - d['hpl'])
            log_prob_if_mws += math.log(1.0 - d['mws'])
    prob_if_eap = math.exp(log_prob_if_eap)
    prob_if_hpl = math.exp(log_prob_if_hpl)
    prob_if_mws = math.exp(log_prob_if_mws)
    if (prob_if_eap + prob_if_hpl + prob_if_mws) > 0:
        divisor = prob_if_eap + prob_if_hpl + prob_if_mws
    else:
        divisor = 1  # the result of the division is zero anyways
    eap = prob_if_eap/divisor
    hpl = prob_if_hpl/divisor
    mws = prob_if_mws/divisor

    probs = {"EAP": eap, "HPL": hpl, "MWS": mws}
    return probs


class NaiveBayesClassifier:
    def __init__(self, k=0.5, matrix={}, filteredWords=[]):
        self.k = k
        self.word_probs = []
        self.matrix = matrix
        self.filteredWords = filteredWords

    def train(self, training_set):
        print("Calculating Num of EAP\n")
        num_eap = len(
            [idx for idx, row in training_set.iterrows() if row.author == "EAP"])
        print("Calculating Num of HPL\n")
        num_hpl = len(
            [idx for idx, row in training_set.iterrows() if row.author == "HPL"])
        print("Calculating Num of MWS\n")
        num_mws = len(
            [idx for idx, row in training_set.iterrows() if row.author == "MWS"])
        print("Counting words\n")
        word_counts = count_words(
            [row for idx, row in training_set.iterrows()], self.matrix, self.filteredWords)
        print("Calculating word Probabilities\n")
        self.word_probs = word_probabilities(
            word_counts, num_eap, num_hpl, num_mws, self.k)

    def classify(self, text):
        return author_probability(self.word_probs, text)


def NinetyFivePercentConfidenceInterval(accuracy, n):
    z = 1.96  # for 95%
    interval = z * sqrt((accuracy * (1 - accuracy)) / n)
    return interval


def SensitivitySpecificity(test_y, predicted_y):
    cnf_matrix = confusion_matrix(test_y, predicted_y)
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    # Sensitivity, hit rate, recall, or true positive rate
    Sensitivity = TP/((TP+FN)*1.0)
    # Specificity or true negative rate
    Specificity = TN/((TN+FP)*1.0)

    return Sensitivity, Specificity


data = pd.read_csv("train.csv")

print("Creating Matrix\n")
M, r, c, filteredWords = createMatrix(data)  # Q1
print("Matrix Creation Done\n")
training_data, testing_data = train_test_split(
    data, test_size=0.33, random_state=42)

#################################### Question 2 #######################################

# for Q2 laplace smoothing constant is zero
print("NaiveBayesClassifier Initialising\n")
classifier = NaiveBayesClassifier(0, M, filteredWords)
print("Classifier Initialising Done\n")

print("Classifier Training\n")
classifier.train(training_data)
print("Classifier Training Done\n")

print("Calculating Author Probabilities\n")
prediction = [
    (row.id,
        classifier.classify(row.text)['EAP'],
        classifier.classify(row.text)['HPL'],
        classifier.classify(row.text)['MWS'],
        row.author
     ) for row in [row for idx, row in testing_data.iterrows()]]

print("Calculating Author Probabilities Done\n")

predict_y = []

print("Predicting Y\n")

for ln in prediction:
    if ln[1] > ln[2]:
        if ln[1] > ln[3]:
            pred = "EAP"
        else:
            pred = "MWS"
    else:
        if ln[2] > ln[3]:
            pred = "HPL"
        else:
            pred = "MWS"

    predict_y.append(pred)

print("Predicting Y done\n")

actual_y = [row.author for row in [
    row for idx, row in testing_data.iterrows()]]

print("Calculating Percentage_Accuracy\n")

Accuracy = accuracy_score(actual_y, predict_y)
Percentage_Accuracy = Accuracy*100
ConfidenceInterval95Percent = NinetyFivePercentConfidenceInterval(
    Accuracy, len(predict_y))
Precision = precision_score(actual_y, predict_y, average='macro')
f1Score = f1_score(actual_y, predict_y, average='macro')
Sensitivity, Specificity = SensitivitySpecificity(actual_y, predict_y)

print("Calculating Percentage_Accuracy Done\n")

#################################### Question 3 #######################################

# for Q3, laplace smoothing constant can be taken as non-zero
print("NaiveBayesClassifier with laplace correction initialising\n")
classifier_laplace = NaiveBayesClassifier(0.33, M, filteredWords)
print("classifier_laplace initialising done\n")

print("classifier_laplace Training\n")
classifier_laplace.train(training_data)
print("classifier_laplace Training done\n")

print("Calculating Author Probabilities laplace\n")
prediction_laplace = [
    (row.id,
        classifier_laplace.classify(row.text)['EAP'],
        classifier_laplace.classify(row.text)['HPL'],
        classifier_laplace.classify(row.text)['MWS'],
        row.author
     ) for row in [row for idx, row in testing_data.iterrows()]]
print("Calculating Author Probabilities laplace done\n")

predict_y_laplace = []

print("Predicting Y with laplace correction\n")

for lnl in prediction_laplace:
    if lnl[1] > lnl[2]:
        if lnl[1] > lnl[3]:
            pred_l = "EAP"
        else:
            pred_l = "MWS"
    else:
        if lnl[2] > lnl[3]:
            pred_l = "HPL"
        else:
            pred_l = "MWS"

    predict_y_laplace.append(pred_l)

print("Predicting Y with laplace correction done\n")

# actual_y = [row.author for row in [
# row for idx, row in testing_data.iterrows()]]

print("Calculating Percentage_Accuracy_laplace\n")

Accuracy_laplace = accuracy_score(actual_y, predict_y_laplace)
Percentage_Accuracy_laplace = Accuracy_laplace*100
ConfidenceInterval95Percent_laplace = NinetyFivePercentConfidenceInterval(
    Accuracy_laplace, len(predict_y_laplace))
Precision_laplace = precision_score(
    actual_y, predict_y_laplace, average='macro')
f1Score_laplace = f1_score(actual_y, predict_y_laplace, average='macro')
Sensitivity_laplace, Specificity_laplace = SensitivitySpecificity(
    actual_y, predict_y_laplace)

print("Calculating Percentage_Accuracy_laplace done\n")

print("Printing to Output File\n")

file = open("./out.txt", "w")

file.write("Solution to Q1\n")
file.write("Matrix, M Created with r = " +
           str(r) + " and c = " + str(c) + "\n\n")

file.write("Solution to Q2 and Q4\n")
file.write("The Accuracy of the Naive Bayes classifier model: " +
           str(Percentage_Accuracy) + "\n")
file.write("The 95% confidence interval of accuracy: " +
           str(ConfidenceInterval95Percent) + "\n")
file.write("The Precision: " +
           str(Precision) + "\n")
file.write("The f-Score: " +
           str(f1Score) + "\n")
file.write("The Sensitivity: " +
           str(Sensitivity) + "\n")
file.write("The Specificity: " +
           str(Specificity) + "\n\n")

file.write("Solution to Q3 and Q4\n")
file.write("The Accuracy of the Naive Bayes classifier model with laplace Correction: " +
           str(Percentage_Accuracy_laplace) + "\n")
file.write("The 95% confidence interval of accuracy with laplace Correction: " +
           str(ConfidenceInterval95Percent_laplace) + "\n")
file.write("The Precision with laplace Correction: " +
           str(Precision_laplace) + "\n")
file.write("The f-Score with laplace Correction: " +
           str(f1Score_laplace) + "\n")
file.write("The Sensitivity with laplace Correction: " +
           str(Sensitivity_laplace) + "\n")
file.write("The Specificity with laplace Correction: " +
           str(Specificity_laplace) + "\n\n")

file.close()
print("Printing to Output File Done\n")
