from sklearn.model_selection import train_test_split
from NeuralNet import *


def main():

    # read datasets
    df = pd.read_csv("../Dataset/input_data.csv", sep=',', index_col=[0])
    labels = pd.read_csv("../Dataset/labels.csv", sep=',', index_col=[0])

    # organize data in images
    data = organizeData(df, len(labels))

    # splitting data in training and testing sets
    train_X, valid_X, train_ground, valid_ground = train_test_split(data, labels, test_size=0.2, random_state=13)

    # create model
    nn = NeuralNet(data[0].shape)

    # train model
    nn.train(np.array(train_X), np.array(train_ground))

    # evaluate model
    nn.evaluate(np.array(valid_X), np.array(valid_ground))

    # save model
    nn.save()


if __name__ == '__main__':
    setUpGPU()
    main()
