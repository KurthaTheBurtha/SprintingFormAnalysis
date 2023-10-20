import numpy as np
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

def train():
    # process inputs
    # inputs are image embeddings generated
    good = np.load('good.npy')
    bad = np.load('bad.npy')

    # target values are 0 for bad form, 1 for good form
    goodtargets = [1]*len(good)
    badtargets = [0]*len(bad)

    # combines good and bad into one sest
    inputs = np.concatenate((good,bad),axis=0)
    target = goodtargets+badtargets

    # split dataset into test and train
    inputs_train, inputs_test, target_train, target_test = train_test_split(inputs,target,test_size=0.5, random_state=42)\

    # fit data on 50% of data
    classifier = MLPClassifier(random_state=0, hidden_layer_sizes=(25,50,10,5),batch_size=5)
    classifier.fit(inputs_train, target_train)

    # get results and print
    results = classifier.predict(inputs_test)
    print(str(round((results == target_test).mean()*100,2))+'% accuracy')


if __name__ == '__main__':
    train()