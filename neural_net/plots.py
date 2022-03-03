from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn


def plotConfusionMatrix(y_test, y_pred, categories):       # plots confusion matrix with percentages
    cm = confusion_matrix(y_test, y_pred)                         # calculating confusion matrix
    cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # calculating percentages of conf matrix

    ax = sn.heatmap(cm_perc, annot=True, fmt='.2f', xticklabels=categories, yticklabels=categories)
    ax.set_title('Confusion Matrix\n')
    ax.set_xlabel('Predicted Category')
    ax.set_ylabel('True Category ')
    plt.show()
    figure = ax.get_figure()
    figure.savefig('../Results/confusion_matrix.png')


def plot_mean_loss(loss, val_loss):  # plots mean loss per epoch
    mean_loss = []
    mean_vloss = []

    for i in range(0, len(loss[0])):
        # calculating mean loss per epoch
        suml = 0
        for lst in loss:
            suml += lst[i]
        mean_loss.append(suml / len(loss))

        # calculating mean validation loss per epoch
        suml = 0
        for lst in val_loss:
            suml += lst[i]
        mean_vloss.append(suml / len(val_loss))

    plt.plot(mean_loss, label='Mean Loss')
    plt.plot(mean_vloss, label='Mean Validation Loss')
    plt.xlabel(r'Epoch')
    plt.ylabel(r'Mean Loss')
    plt.legend(loc='upper right')
    plt.show()
    plt.savefig("../Results/mean_loss.jpg")


def plot_mean_accuracy(acc, val_acc):  # plots mean accuracy per epoch
    mean_acc = []
    mean_vacc = []

    for i in range(0, len(acc[0])):
        # calculating mean accuracy per epoch
        suma = 0
        for lst in acc:
            suma += lst[i]
        mean_acc.append(suma / len(acc))

        # calculating mean validation accuracy per epoch
        suma = 0
        for lst in val_acc:
            suma += lst[i]
        mean_vacc.append(suma / len(val_acc))

    plt.plot(mean_acc, label='Mean Accuracy')
    plt.plot(mean_vacc, label='Mean Validation Accuracy')
    plt.xlabel(r'Epoch')
    plt.ylabel(r'Mean Accuracy')
    plt.legend(loc='lower right')
    plt.show()
    plt.savefig("../Results/mean_accuracy.jpg")


def plot_loss_per_fold(loss, val_loss):    # plots loss and val loss for every fold
    count = 1

    for ls, vls in zip(loss, val_loss):
        plt.plot(ls, label='Loss - Fold ' + str(count))
        plt.plot(vls, label='Validation Loss - Fold ' + str(count))
        plt.xlabel(r'Epoch')
        plt.ylabel(r'Loss')
        plt.legend(loc='upper right')
        count += 1

    plt.show()
    plt.savefig("../Results/fold_loss.jpg")


def plot_accuracy_per_fold(acc, val_acc):  # plots accuracy and val accuracy for every fold
    count = 1

    for ac, vac in zip(acc, val_acc):
        plt.plot(ac, label='Accuracy - Fold ' + str(count))
        plt.plot(vac, label='Validation Accuracy - Fold ' + str(count))
        plt.xlabel(r'Epoch')
        plt.ylabel(r'Accuracy')
        plt.legend(loc='lower right')
        count += 1

    plt.show()
    plt.savefig("../Results/fold_accuracy.jpg")
