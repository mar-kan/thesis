from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow import keras
# from numpy import mod
import pandas as pd
import time

from functions import *
from plots import *


class NeuralNet:
    model = None
    input_shape = 0

    num_conv_layers = 3
    filters = 64
    filter_size = 4

    folds = 15
    epochs = 250
    batch_size = 128

    categories = ["ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA", "GBM", "HNSC", "KICH",
                  "KIRC", "KIRP", "LAML", "LGG", "LIHC", "LUAD", "LUSC", "MESO", "OV", "PAAD", "PCPG",
                  "PRAD", "READ", "SARC", "SKCM", "STAD", "TGCT", "THCA", "THYM", "UCEC", "UCS", "UVM"]

    def __init__(self, shape):  # creates model
        self.model = Sequential()
        self.input_shape = shape

        # creating model layers
        for i in range(0, self.num_conv_layers):
            if i == 0:
                self.model.add(layers.Conv2D(self.filters, (self.filter_size, self.filter_size), strides=(1, 1),
                                             padding="valid", activation='relu',
                                             input_shape=(self.input_shape[0], self.input_shape[1], 1)))
            else:
                self.model.add(layers.Conv2D(self.filters, (self.filter_size, self.filter_size), strides=(1, 1),
                                             padding="valid", activation='relu'))

            self.model.add(layers.MaxPooling2D((2, 2), strides=2))
            self.model.add(layers.BatchNormalization(epsilon=1e-05, momentum=0.1))

            self.filters *= 2
            self.filter_size = int(self.filter_size / 2)
            # if mod(self.filter_size, 2) != 0:
            #     self.filter_size += 1

        self.model.add(layers.Dropout(rate=0.25))
        self.model.add(layers.Flatten())

        self.model.add(layers.Dense(1024, use_bias=True, activation='relu'))
        self.model.add(layers.Dense(512, use_bias=True, activation='relu'))
        self.model.add(layers.Dense(32, use_bias=True, activation='sigmoid'))

        # compiling model
        self.model.compile(optimizer=keras.optimizers.Adamax(learning_rate=0.0001),
                           loss='categorical_crossentropy', metrics=['accuracy'])

        self.model.summary()

    def train(self, X, Y):  # trains and evaluates model
        start = time.time()

        acc = []
        loss = []
        val_acc = []
        val_loss = []
        all_pred = []
        all_test = []
        count = 1

        # k cross validation
        kf = StratifiedKFold(n_splits=self.folds, shuffle=True)
        for train, test in kf.split(X, Y):
            print("Fold " + str(count) + ":")

            # converting labels to one hot encodings
            train_categories = oneHotEncode(Y[train])
            test_categories = oneHotEncode(Y[test])

            # training model
            history = self.model.fit(X[train], train_categories, epochs=self.epochs, batch_size=self.batch_size,
                                     verbose=1, validation_data=(X[test], test_categories))

            # storing values for model evaluation
            acc.append(history.history['accuracy'])
            loss.append(history.history['loss'])
            val_acc.append(history.history['val_accuracy'])
            val_loss.append(history.history['val_loss'])

            all_pred.append(self.getPredictions(X[test]))
            all_test.append(np.argmax(test_categories, axis=1))
            count += 1

        end = time.time()
        print("Training time:" + str(end - start) + " seconds.")

        # evaluate model
        self.evaluateTraining(all_test, all_pred, acc, val_acc, loss, val_loss)

    def evaluateTraining(self, test, pred, acc, val_acc, loss, val_loss):  # evaluates training
        print("\n\nResults of validation sets during training.\n")

        # plot loss and accuracy for all folds
        plot_accuracy_per_fold(acc, val_acc)
        plot_loss_per_fold(loss, val_loss)

        # plot mean loss and accuracy
        plot_mean_accuracy(acc, val_acc)
        plot_mean_loss(loss, val_loss)

        # flatten lists
        test = [item for sublist in test for item in sublist]
        pred = [item for sublist in pred for item in sublist]

        # printing classification report for all predictions
        print("Classification Report")
        print(classification_report(test, pred))

        print("Mean loss: %.3f" % np.mean(val_loss))
        print("Mean accuracy: %.3f\n\n" % np.mean(val_acc))

    def evaluate(self, X_eval, Y_eval):  # evaluates model
        print("Results of test set.\n")

        pred_eval = self.getPredictions(X_eval)

        # converting labels to one hot encodings and then to single digit
        eval_categories = oneHotEncode(Y_eval)
        sd = np.argmax(eval_categories, axis=1)

        print("Classification Report")
        print(classification_report(sd, pred_eval))

        # saving classification report to csv
        report = classification_report(sd, pred_eval, output_dict=True)
        report = pd.DataFrame(report).transpose()
        report = report.set_axis(self.categories + ['accuracy', 'macro avg', 'weighted avg'], axis='index')
        report.to_csv('../results/classification_report.csv', sep=',')

        # plotting confusion matrix
        plotConfusionMatrix(sd, pred_eval, self.categories)

        results = self.model.evaluate(X_eval, eval_categories)
        print("Mean loss: %.3f" % results[0])
        print("Mean accuracy: %.3f" % results[1])

    def getPredictions(self, X):     # get model's predictions
        pred = self.model.predict(X, batch_size=self.batch_size, verbose=1)
        return np.argmax(pred, axis=1)

    def save(self):     # saves model
        self.model.save(os.getcwd() + '/SavedModel/model.h5', overwrite=True)
        self.model.save_weights(os.getcwd() + '/SavedModel/weights.hdf5', overwrite=True)
        # self.model.summary()
