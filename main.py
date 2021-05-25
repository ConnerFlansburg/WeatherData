"""
main.py creates, trains, and tests machine learning models when provided with test & training data sets.

Authors/Contributors: Dr. Dimitrios Diochnos, Conner Flansburg

Github Repo:
"""
# import logging as log
# import traceback
# import typing as typ
from formatting import printPercentage, printWarn, printUnderline, colorDecimal
import pathlib
import pandas as pd
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
import sys
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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
    """ Parses the incoming CSV. """
    # * Read in the Data * #
    data = pd.read_csv(filename)  # read the data into a panda dataframe

    # * Turn -1 & 1 into String Values * #
    # data['S1'].replace(1, 'Tornado', inplace=True)
    # data['S1'].replace(-1, 'No Tornado', inplace=True)

    # * Print Info about Data * #
    if print_data:

        freq = data["S1"].value_counts()
        print(freq)

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
    # dummy_model: DummyClassifier = DummyClassifier(strategy="constant", constant="No Tornado")
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

    # ****************************** Report Results ****************************** #
    VERBOSE_ZERO = 0    # used by the longer (verbose) report

    # * Dummy Model * #  (this is module always guesses No Tornado)
    print(f'\n{"Dummy Model Report - No Tornado":^59}')

    printStats(test_labels, prediction_dummy, dummy_score)
    if VERBOSE_REPORT:
        names = ['No Tornado', 'Tornado']  # 1 = Tornado, -1 = No Tornado (this is used to alias names in the report)
        report = classification_report(test_labels, prediction_dummy, zero_division=VERBOSE_ZERO, target_names=names)
        print(f'\n{"SciKit Report":^59}\n{report}')

    # * SVC Model * #
    print(f'\n{"SVC Model Report":^59}')
    printStats(test_labels, prediction, svc_score)
    if VERBOSE_REPORT:
        names = ['No Tornado', 'Tornado']  # 1 = Tornado, -1 = No Tornado (this is used to alias names in the report)
        report = classification_report(test_labels, prediction, zero_division=VERBOSE_ZERO, target_names=names)
        print(f'\n{"SciKit Report":^59}\n{report}')


def printStats(true_labels, predicted_labels, score):
    """ Creates a report using various statistics & print the result to the console."""

    # Get the True Negatives, False Positives, False Negatives, & True Positives
    TN, FP, FN, TP = confusion_matrix(true_labels, predicted_labels).ravel()

    row_1 = f"{f'True Positives = {TP}':^25} {'||':^4} {f'False Negatives = {FN}':^25}"
    row_2 = f"{f'False Positives = {FP}':^25} {'||':^4} {f'True Negatives = {TN}':^25}"

    # * Perform the Calculations & Handle Divide by 0 Case * #
    # Precision Calculation
    denom = TP + FP
    if denom == 0:  # if we would divide by zero,
        P = 0.0     # then set precision to zero
        col_1 = f"{printUnderline('Precision')}: {colorDecimal(P)} {printWarn('(NaN)')}"
        col_1 = f"{col_1:^56}"
    else:           # otherwise perform calculation
        P = round(TP/(TP + FP), 3)
        col_1 = f"{printUnderline('Precision')}: {colorDecimal(P)}"
        col_1 = f"{col_1:^46}"

    # Recall Calculation
    denom = TP + FN
    if denom == 0:  # if we would divide by zero,
        R = 0.0     # then set precision to zero
        col_2 = f"{printUnderline('Recall')}: {colorDecimal(R)} {printWarn('(NaN)')}"
        col_2 = f"{col_2:^56}"
    else:  # otherwise perform calculation
        R = round(TP / (TP + FN), 3)
        col_2 = f"{printUnderline('Recall')}: {colorDecimal(R)}"
        col_2 = f"{col_2:^46}"

    row_3 = f"{col_1} {'||':^4} {col_2}"

    # F1 Score Calculation
    denom = P + R
    if denom == 0:  # if we would divide by zero,
        F1 = 0.0    # then set precision to zero
        col_1 = f"{printUnderline('F1 Score')}: {colorDecimal(F1)} {printWarn('(NaN)')}"
        col_1 = f"{col_1:^56}"  # center row 4
    else:  # otherwise perform calculation
        F1 = round((2 * ((P * R) / (P + R))), 3)
        col_1 = f"{printUnderline('F1 Score')}: {colorDecimal(F1)}"
        col_1 = f"{col_1:^46}"  # center row 4

    col_2 = f"{printUnderline('Accuracy Score')}: {printPercentage(score)}"  # grab the accuracy of the model
    row_4 = f"{col_1} {'||':^4} {col_2:^35}"

    # get the longest row
    size = 57
    print('=' * size)
    print(row_1)
    print('=' * size)
    print(row_2)
    print('=' * size)
    print(row_3)
    print('=' * size)
    print(row_4)
    print('=' * size)


if __name__ == '__main__':
    main()
