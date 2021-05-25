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

# TODO: get/set from arguments
VERBOSE_REPORT = True  # VERBOSE_REPORT is a bool that determines if the longer report should also be created

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
        # * Turn -1 & 1 into String Values * #
        data['S1'].replace(1, 'Tornado', inplace=True)
        data['S1'].replace(-1, 'No Tornado', inplace=True)

        freq = data["S1"].value_counts()
        print(freq)

        # * Change Back * #
        data['S1'].replace('Tornado', 1, inplace=True)
        data['S1'].replace('No Tornado', -1, inplace=True)

    # ! Debugging/Testing ! #
    # printError("Printing Dataframe...")
    # print(f"{data}\n")
    # !!!!!!!!!!!!!!!!!!!!! #
    
    # * Remove the N Columns * #
    # ? What are N1-N4?
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
    SYSOUT.write(HDR + 'Testing SVC Model...\n')
    
    test_labels, ftrs = get_Label_and_Features(test_filename, True)

    prediction_dummy = dummy_model.predict(ftrs)  # make the dummy prediction
    dummy_score: float = accuracy_score(test_labels, prediction_dummy)  # test the prediction

    prediction = SVC_model.predict(ftrs)  # make prediction
    svc_score = accuracy_score(test_labels, prediction)  # test prediction
    
    SYSOUT.write(OVERWRITE + ' SVC Model Tested '.ljust(50, '-') + SUCCESS)

    # ****************************** Report Calculations ****************************** #
    # *            TP = True Positive              FP = False Positive                * #
    # *            TN = True Negative              FN = False Negative                * #
    # *            Precision = TP/(TP + FP)        Recall = TP/(TP+FN)                * #
    # ********************************************************************************* #
    # * For details on Scikit Calculations See: https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
    # What value should be used on a divide by zero? (these are constants & must be 0, 1, or 'warn')
    PRECISION_ZERO = 0  # value used during precision calc
    RECALL_ZERO = 0     # value used during recall calc
    VERBOSE_ZERO = 0    # used by the longer (verbose) report

    # * Dummy Model * #  (this is module always guesses No Tornado)
    print(f"\nDummy Model Report - No Tornado")
    printDecimal(dummy_score, 'Accuracy')  # print the accuracy of the dummy model
    printDecimal(precision_score(test_labels, prediction_dummy, zero_division=PRECISION_ZERO), 'Precision')
    printDecimal(recall_score(test_labels, prediction_dummy, zero_division=RECALL_ZERO), 'Recall')
    if VERBOSE_REPORT:
        names = ['Tornado', 'No Tornado']  # 1 = No Tornado, -1 = Tornado (this is used to alias names in the report)
        report = classification_report(test_labels, prediction_dummy, zero_division=VERBOSE_ZERO, target_names=names)
        print(f"Dummy Model Verbose Report\n{report}")

    # * SVC Model * #
    print(f"\nSVC Model Report - No Tornado")
    printDecimal(svc_score, 'Accuracy')  # print the accuracy of the created model
    printDecimal(precision_score(test_labels, prediction, zero_division=PRECISION_ZERO), 'Precision')
    printDecimal(recall_score(test_labels, prediction, zero_division=RECALL_ZERO), 'Recall')
    if VERBOSE_REPORT:
        names = ['Tornado', 'No Tornado']  # 1 = No Tornado, -1 = Tornado (this is used to alias names in the report)
        report = classification_report(test_labels, prediction, zero_division=VERBOSE_ZERO, target_names=names)
        print(f"SVC Model Verbose Report\n{report}")


def printDecimal(decimalScore: float, msg: str):

    if decimalScore > 0.75:  # > 75 print in green
        SYSOUT.write(f'\r\033[32;1m{msg} Score: {round(decimalScore * 100, 2)}%\033[00m\n')
        SYSOUT.flush()

    elif 0.45 < decimalScore < 0.75:  # > 45 and < 75 print yellow
        SYSOUT.write(f'\r\033[33;1m{msg} Score: {round(decimalScore * 100, 2)}%\033[00m\n')
        SYSOUT.flush()

    elif decimalScore < 0.45:  # < 45 print in red
        SYSOUT.write(f'\r\033[91;1m{msg} Score: {round(decimalScore * 100, 2)}%\033[00m\n')
        SYSOUT.flush()

    else:  # don't add color, but print accuracy
        SYSOUT.write(f'{msg} Score: {decimalScore}\n')
        SYSOUT.flush()


def printAccuracy(percentScore: float):

    if percentScore > 75:  # > 75 print in green
        SYSOUT.write(f'\r\033[32;1m Accuracy Score: {percentScore}%\033[00m\n')
        SYSOUT.flush()

    elif 45 < percentScore < 75:  # > 45 and < 75 print yellow
        SYSOUT.write(f'\r\033[33;1m Accuracy Score: {percentScore}%\033[00m\n')
        SYSOUT.flush()

    elif percentScore < 45:  # < 45 print in red
        SYSOUT.write(f'\r\033[91;1m Accuracy Score: {percentScore}%\033[00m\n')
        SYSOUT.flush()

    else:  # don't add color, but print accuracy
        SYSOUT.write(f' Accuracy is: {percentScore}%\n')
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
