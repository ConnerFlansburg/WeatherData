"""
main.py creates, trains, and tests machine learning models when provided with test & training data sets.

Authors/Contributors: Dr. Dimitrios Diochnos, Conner Flansburg

Github Repo:
"""
import pathlib
# import typing as typ
import pandas as pd
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
import sys
# import logging as log
# import traceback
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
from pyfiglet import Figlet


SEED: int = 498
HDR = '*' * 6
SUCCESS = u' \u2713\n'+'\033[0m'     # print the checkmark & reset text color
OVERWRITE = '\r' + '\033[32;1m' + HDR  # overwrite previous text & set the text color to green
NO_OVERWRITE = '\033[32;1m' + HDR      # NO_OVERWRITE colors lines green that don't use overwrite
SYSOUT = sys.stdout

# TODO: Compare to the version that just always says no tornado
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


def get_Label_and_Features(filename: str, print_data: bool = False):
    
    # * Read in the Data * #
    data = pd.read_csv(filename)  # read the data into a panda dataframe

    # * Print Info about Data * #
    if print_data:
        freq = data["S1"].value_counts()
        print("Values for S1 in data (1 = tornado, -1 = no tornado):")
        print(freq)

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
    labels, ftrs = get_Label_and_Features(training_filename, True)

    # * Create the SVC Model * #
    SYSOUT.write(HDR + 'Creating SVC Model...')
    
    SVC_model: SVC = SVC(kernel='sigmoid', random_state=SEED)
    SVC_model.fit(ftrs, labels)  # train the model

    SYSOUT.write(OVERWRITE + ' SVC Model Created '.ljust(50, '-') + SUCCESS)

    # * Create the Dummy Model * #
    dummy_model: DummyClassifier = DummyClassifier(strategy="constant", constant=-1)
    dummy_model.fit(ftrs, labels)

    # * Test the Model * #
    SYSOUT.write(HDR + 'Testing SVC Model...')
    
    test_labels, ftrs = get_Label_and_Features(test_filename, True)

    prediction_dummy = dummy_model.predict(ftrs)  # make the dummy prediction
    dummy_score = accuracy_score(test_labels, prediction_dummy)  # test the prediction
    dType: str = 'Dummy Predictor'

    prediction_score = SVC_model.predict(ftrs)  # make prediction
    score = accuracy_score(test_labels, prediction_score)  # test prediction
    mType: str = 'SVC'
    
    SYSOUT.write(OVERWRITE + ' SVC Model Tested '.ljust(50, '-') + SUCCESS)

    # * Report Result * #
    percentScore: float = round(score * 100, 1)  # turn the score into a percent with 2 decimal places
    dummyScore: float = round(dummy_score * 100, 1)
    printAccuracy(percentScore, mType)  # print the accuracy of the created model
    printAccuracy(dummyScore, dType)  # print the accuracy of the dummy model

    print(f"Dummy Model Report")
    print(f"Precision score: {precision_score(test_labels, prediction_dummy)}")
    print(f"Recall Score: {recall_score(test_labels, prediction_dummy)}\n")

    print(f"SVC Model Report")
    print(f"Precision score: {precision_score(test_labels, prediction_score)}")
    print(f"Recall Score: {recall_score(test_labels, prediction_score)}")

    print(f"Dummy Model Report\n{classification_report(test_labels, prediction_dummy)}")
    print(f"SVC Model Report\n{classification_report(test_labels, prediction_score)}")



def printAccuracy(percentScore: float, mType: str):

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
