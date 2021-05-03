"""
main.py creates, trains, and tests machine learning models when provided with test & training data sets.

Authors/Contributors: Dr. Dimitrios Diochnos, Conner Flansburg

Github Repo:
"""
import pathlib
# import typing as typ
# import numpy as np
import pandas as pd
from sklearn.svm import SVC
import sys
# import logging as log
# import traceback
from sklearn.metrics import accuracy_score
from pyfiglet import Figlet


SEED: int = 498
HDR = '*' * 6
SUCCESS = u' \u2713\n'+'\033[0m'     # print the checkmark & reset text color
OVERWRITE = '\r' + '\033[32;1m' + HDR  # overwrite previous text & set the text color to green
NO_OVERWRITE = '\033[32;1m' + HDR      # NO_OVERWRITE colors lines green that don't use overwrite
SYSOUT = sys.stdout

# TODO: Add documentation


def main() -> None:
    
    # * Start Up * #
    title: str = Figlet(font='larry3d').renderText('Weather Data')
    SYSOUT.write(f'\033[34;1m{title}\033[00m')  # formatted start up message
    SYSOUT.write("\033[32;1mProgram Initialized Successfully\033[00m\n")

    # * Select Training Data * #
    trainPath = pathlib.Path.cwd() / 'data' / 'training_tornado.csv'  # create a Path object
    training_filename = str(trainPath)

    # * Select Testing Data * #
    testPath = pathlib.Path.cwd() / 'data' / 'testing_tornado.csv'  # create a Path object
    test_filename = str(testPath)

    # * Train & Test the Model * #
    train_and_test(training_filename, test_filename)  # train & test the model(s)


def get_Label_and_Features(filename: str):
    
    # * Read in the Data * #
    data = pd.read_csv(filename)  # read the data into a panda dataframe

    # ! Debugging/Testing ! #
    # printError("Printing Dataframe...")
    # print(f"{data}\n")
    # !!!!!!!!!!!!!!!!!!!!! #
    
    # * Remove the N Columns * #
    del data["N1"]
    del data["N2"]
    del data["N3"]
    del data["N4"]
    
    # * Get the Features * #
    ftrs = data.drop("S1", axis=1)
    
    # * Get the Labels * #
    labels = data["S1"]

    # ! Debugging/Testing ! #
    # printError("Printing Features...")
    # print(f"{ftrs}\n")
    # printError("Printing Labels...")
    # print(f"{labels}\n")
    # !!!!!!!!!!!!!!!!!!!!! #
    
    return labels, ftrs


def train_and_test(training_filename: str, test_filename: str):

    # * Get the Labels & Features from the Training Data
    labels, ftrs = get_Label_and_Features(training_filename)

    # * Create the SVC Model * #
    SYSOUT.write(HDR + 'Creating SVC Model...')
    
    SVC_model: SVC = SVC(kernel='sigmoid', random_state=SEED)
    SVC_model.fit(ftrs, labels)  # train the model
    
    SYSOUT.write(OVERWRITE + ' SVC Model Created '.ljust(50, '-') + SUCCESS)

    # * Test the Model * #
    SYSOUT.write(HDR + 'Testing SVC Model...')
    
    test_labels, ftrs = get_Label_and_Features(test_filename)
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
