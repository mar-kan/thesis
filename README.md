This project uses RNA-sequences, converted in 2D images, to classify cancer types using Deep Learning. 

It is divided in three parts. The first one is written in R and is used to download the data and store them. 

The second one is written in Python and is used to preprocess the data and convert the RNA-sequences to 2D images.
The third part is written in Python and it is the part where the classification takes place via a Convolutional Neural Network.

All data used were downloaded from the TCGA project (https://portal.gdc.cancer.gov/), using the R library TCGA biolinks. 

----------------------
## Part 1 

This part of the project is written in R. It is used to download the TCGA data. It outputs a dataset for each cancer type, 
adding a row with its label.

### Execution instructions: 

    Before the first execution, execute the "set-up-packages.R" script to set up the environment and install all 
    R packages required. R 4.1 must have already been installed.

    execute "main.R"

### Documentation:

**set-up-packages.R:** script that installs all the prerequisite libraries needed for the program to run.

**functions.R:** contains functions used to download and prepare TCGA data using TCGAbiolinks. 
    Includes the following functions:

    loadLibraries: loads all libraries needed for the program to run.

    downloadData: downloads and prepares all data needed for gene expression using GDCquery.
        arguments: <project>: string that includes the code name of the TCGA project to be downloaded
        return value: returns a ranged summarized experiment including all the downloaded data


**main.R:** Script main downloads and prepares 33 datasets from TCGA.

----------------------

## Part 2

This part of the project is written in Python. It is used to preprocess the TCGA data, convert the samples to images 
and finally output one large dataset with two columns: images and labels.

### Execution instructions: 
    
    jupyter notebook preprocess.ipynb

### Documentation:

**Data.py:** Contains the class <Data> that stores the data and their labels and contains methods for the data 
preprocessing. Also contains functions used for the preprocessing and reading of the data.

Contains the following methods:

    Constructor: initializes tha dataframe
        arguments: <data>: a Dataframe that contains the input data
    
    normalizeData: normalizes one gene sample and scales it to remove the noise, using the log transform y = log2(x+1)
        arguments: <x>: numeric value that represents one of the gene's values
        return value: returns the result of the normalization of <x>

    sortByChromosome: sorts the dataframe by chromosome number

    geneFiltering: filters out of the dataframe the genes that don't change much between samples 
        (their variance is < 1.3)
        
    reshapeData: reshapes each gene to a 2D image of the nearest square number. Adds zeros to fill the values of 
        the image
        return value: returns an array that contains all the images


Contains the following functions:

    loadDataset: loads all datasets in the directory given in a dataframe
        arguments: <dir_path>: the path of the directory to read
        return value: returns the dataframe containing each file's data separetely

    mergeDatasets: merges the data of each file of the above dataframe to one
        arguments: <datasets>: the dataframe containing each file's data separetely
        return value: returns the merged dataframe

    scaleReduction: reduces the scale of a gene using a logarithmic transformation
        arguments: <elem>: a gene
        return value: returns the scaled gene

    noiseReduction: reduces the noise of a gene whose value is close to zero
        arguments: <elem>: a gene
        return value: if the gene's value is less than 1, returns 0
                      otherwise returns the gene

    imageNormalization: normalizes an image's value (dividing it by 255)
        arguments: <elem>: an image value
        return value: returns the normalized value

**preprocess.ipynb:** Preprocesses the data and converts them to 2D images, using the above class and functions. 
Prints an example image of each dataset. Finally, exports a csv with all the images and a csv with their labels.


----------------------

## Part 3

This part of the project is written in Python with Tensorflow and Keras. It contains a CNN that is used to classify the cancer types.

### Execution instructions: 
    
    python3 neuralNetworkMain.py

### Documentation:

**functions.py:** Includes various general functions used in the project. 
These functions are the following:

    setUpGPU: sets up the configurations for running with GPU

    organizeData: splits the dataset to images and reshapes them to 108x108x1
        arguments: <df>: the dataset read of the file
                   <num>: the number of the images
        return value: returns the organized dataset
    
    oneHotEncode: one hot encodes the labels of the dataset
        arguments: <labels>: the labels of the dataset
        return value: the one hot encodings of the labels

**NeuralNet.py:** Class for the neural network. Stores the model, its results and its hyperparameters.
Contains the following methods:

    Constructor: initializes the input size and the model. Creates its layers and compiles it.
        arguments: <shape>: contains the shape of the images [n x n]
    
    train: trains the model using StratifiedKFold.
        arguments: <X>: input dataset
                   <Y>: label set

    plotTraining: plots loss and accuracy from training and prints its classification report
        arguments: <test>: the validation set from the whole training
                   <pred>: the predictions of the training
                   <acc>: the accuracy  of the training
                   <val_acc>: the validation accuracy of the training
                   <loss>: the loss of the training
                   <val_loss>: the validation loss of the training

    evaluate: evaluates the model using plots from the training, its confusion matrix and sklearn's classification 
        report. Plots confusion matrix using the accuracy of its results.
        arguments: <X>: evaluation input dataset
                   <Y>: evaluation label set

    getPredictions: calculates the predictions of the model for the given input set
        arguments: <X>: subset of input dataset
        return value: returns the predictions it calculated 

    save: saves the model and its weights in files.

**plots.py:** Creates functions to plot a confusion matrix and loss and accuracy in various ways.

Contains the following functions:

    plotConfusionMatrix: plots the confusion matrix for the given predictions and labels
        arguments: <y_test>: labels with true results
                   <y_pred>: predictions of the model

    plot_mean_loss: calculates mean loss and mean validation loss per epoch for all folds and plots it
        arguments: <loss>: the loss of the training
                   <val_loss>: the validation loss

    plot_mean_accuracy: calculates mean accuracy and mean validation accuracy per epoch for all folds and plots it
        arguments: <loss>: the accuracy of the training
                   <val_loss>: the validation accuracy

    plot_loss_per_fold: plots loss and validation loss of model for all epochs
        arguments: <loss>: the loss of the training
                   <val_loss>: the validation loss

    plot_accuracy_per_fold: plots accuracy and validation accuracy of model for all epochs
        arguments: <acc>: the accuracy of the training
                   <val_acc>: the validation accuracy


**main.py:** Creates and runs the neural network with the csvs that were exported from <preprocess.ipynb>. Trains it, 
evaluates it and saves it.






