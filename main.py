"""
main.py creates, trains, and tests machine learning models when provided with test & training data sets.

Authors/Contributors: Dr. Dimitrios Diochnos, Conner Flansburg

Github Repo:
"""
import pathlib as path
import typing as typ
import numpy as np
import pandas as pd
from sklearn.svm import SVC
import sys
import logging as log
import traceback
from sklearn.metrics import accuracy_score
from pyfiglet import Figlet


SEED: int = 498
HDR = '*' * 6
SUCCESS = u' \u2713\n'+'\033[0m'     # print the checkmark & reset text color
OVERWRITE = '\r' + '\033[32;1m' + HDR  # overwrite previous text & set the text color to green
NO_OVERWRITE = '\033[32;1m' + HDR      # NO_OVERWRITE colors lines green that don't use overwrite
SYSOUT = sys.stdout

# TODO: Add documentation


def main(training_filename: str, test_filename: str) -> None:

    title: str = Figlet(font='larry3d').renderText('Weather Data')
    SYSOUT.write(f'\033[34;1m{title}\033[00m')  # formatted start up message
    SYSOUT.write("\033[32;1mProgram Initialized Successfully\033[00m\n")

    train_and_test(training_filename, test_filename)  # train & test the model(s)


def train_and_test(training_filename: str, test_filename: str):

    # * Read the Two CSV Files into Dataframes * #
    SYSOUT.write(HDR + 'Reading in CSVs...')
    training: np.ndarray = np.genfromtxt(training_filename, delimiter=',', skip_header=1)
    testing: np.ndarray = np.genfromtxt(test_filename, delimiter=',', skip_header=1)
    SYSOUT.write(OVERWRITE + ' CSVs Parsed '.ljust(50, '-') + SUCCESS)

    # * Get the Labels & Features from the Training Data
    ftrs, labels = formatForSciKit(training)

    # * Create the SVC Model * #
    SYSOUT.write(HDR + 'Creating SVC Model...')
    SVC_model: SVC = SVC(kernel='sigmoid', random_state=SEED)
    SVC_model.fit(ftrs, labels)  # train the model
    SYSOUT.write(OVERWRITE + ' SVC Model Created '.ljust(50, '-') + SUCCESS)

    # * Test the Model * #
    SYSOUT.write(HDR + 'Testing SVC Model...')
    ftrs, test_labels = formatForSciKit(testing)

    prediction_score = SVC_model.predict(ftrs)  # make prediction
    score = accuracy_score(test_labels, prediction_score)  # test prediction
    mType: str = 'SVC'
    SYSOUT.write(OVERWRITE + ' SVC Model Tested '.ljust(50, '-') + SUCCESS)

    # * Report Result * #
    percentScore: float = round(score * 100, 1)  # turn the score into a percent with 2 decimal places

    if percentScore > 75:  # > 75 print in green
        SYSOUT.write(f'\r\033[32;1m{mType} Accuracy is: {percentScore}%\033[00m\n')
        SYSOUT.flush()

    elif 45 < percentScore < 75:  # > 45 and < 75 print yellow
        SYSOUT.write(f'\r\033[33;1m{mType} Accuracy is: {percentScore}%\033[00m\n')
        SYSOUT.flush()

    elif percentScore < 45:  # < 45 print in red
        SYSOUT.write(f'\r\033[91;1m{mType} Accuracy is: {percentScore}%\033[00m\n')
        SYSOUT.flush()

    else:  # don't add color, but print accuracy
        SYSOUT.write(f'{mType} Accuracy is: {percentScore}%\n')
        SYSOUT.flush()


def formatForSciKit(data: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    formatForSciKit takes the input data and converts it into a form that can
    be understood by the sklearn package. It does this by separating the features
    from their labels and returning them as two different numpy arrays.

    :param data: The input data, from a read in CSV.
    :type data: np.ndarray

    :return: The input file in a form parsable by sklearn.
    :rtype: tuple[np.ndarray, np.ndarray]
    """

    # create the label array Y (the target of our training)
    # from all rows, pick the 0th column
    try:
        # + data[:, :1] get every row but only the first column
        flat = np.ravel(data[:, :1])  # get a list of all the labels as a list of lists & then flatten it
        labels = np.array(flat)  # convert the label list to a numpy array
        # create the feature matrix X ()
        # + data[:, 1:] get every row but drop the first column
        ftrs = np.array(data[:, 1:])  # get everything BUT the labels/ids

    except (TypeError, IndexError) as err:
        lineNm = sys.exc_info()[-1].tb_lineno  # get the line number of error
        msg = f'{str(err)}, line {lineNm}:\ndata = {data}\ndimensions = {data.ndim}'
        log.error(msg)  # log the error
        printError(msg)  # print message
        traceback.print_stack()  # print stack trace
        sys.exit(-1)  # exit on error; recovery not possible

    return ftrs, labels


def printError(message: str) -> None:
    """
    printError is used for coloring error messages red.

    :param message: The message to be printed.
    :type message: str

    :return: printError does not return, but rather prints to the console.
    :rtype: None
    """
    print("\033[91;1m {}\033[00m".format(message))

if __name__ == '__main__':
    main()

