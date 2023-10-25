import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier

def train():
    # process inputs
    # inputs are image embeddings generated
    good = np.load('good.npy')
    bad = np.load('bad.npy')

    # target values are 0 for bad form, 1 for good form
    goodtargets = [1] * len(good)
    badtargets = [0] * len(bad)

    # combines good and bad into one sest
    inputs = np.concatenate((good, bad), axis=0)
    target = goodtargets + badtargets

    # split dataset into test and train
    inputs_train, inputs_test, target_train, target_test = train_test_split(inputs, target, test_size=0.5, random_state=42)

    # # 96.67% accuracy
    # classifier = MLPClassifier(random_state=0, hidden_layer_sizes=(25, 50, 10, 5), batch_size=5)
    # classifier.fit(inputs_train, target_train)
    #
    # results = classifier.predict(inputs_test)
    # print(str(round((results == target_test).mean() * 100, 2)) + '% accuracy')

    model = MLPClassifier()
    grid = GridSearchCV(model, {
        'random_state': [0],
        'hidden_layer_sizes': [(50,) * num_layers for num_layers in range(1, 5)],
        'batch_size': [5, 10, 20, 50],
        'learning_rate_init': [0.001, ]
    }, verbose=1)
    grid.fit(inputs_train, target_train)
    results = grid.score(inputs_test,target_test)
    print(results)
    # breakpoint()

if __name__ == '__main__':
    train()
