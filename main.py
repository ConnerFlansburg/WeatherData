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
from sklearn.metrics import accuracy_score, confusion_matrix
from pyfiglet import Figlet

SEED: int = 498
HDR = '*' * 6
SUCCESS = u' \u2713\n'+'\033[0m'       # print the checkmark & reset text color
OVERWRITE = '\r' + '\033[32;1m' + HDR  # overwrite previous text & set the text color to green
NO_OVERWRITE = '\033[32;1m' + HDR      # NO_OVERWRITE colors lines green that don't use overwrite
SYSOUT = sys.stdout                    # SYSOUT set the standard out for the program to the console
FREQUENCY = None

# TODO: get/set from arguments
VERBOSE_REPORT = False  # VERBOSE_REPORT is a bool that determines if the longer report should also be created

# TODO: Compare to the version that just always says no tornado
# TODO: Add documentation


def main() -> None:
    
    # * Start Up * #
    title: str = Figlet(font='larry3d').renderText('Weather Data')
    SYSOUT.write(f'\033[34;1m{title}\033[00m')  # formatted start up message

    # * Select Training Data * #
    trainPath = pathlib.Path.cwd() / 'data' / 'training_tornado.csv'  # create a Path object
    training_filename = str(trainPath)

    # * Select Testing Data * #
    testPath = pathlib.Path.cwd() / 'data' / 'testing_tornado.csv'  # create a Path object
    test_filename = str(testPath)

    # * Train & Test the Model * #
    train_and_test(training_filename, test_filename)  # train & test the model(s)


def get_Label_and_Features(input_df):
    """ Parses the incoming CSV. (data should be a Pandas dataframe) """

    # * Remove the N Columns * #
    # ? What are N1-N4?
    del input_df["N1"]
    del input_df["N2"]
    del input_df["N3"]
    del input_df["N4"]
    
    # * Get the Features * #
    ftrs = input_df.drop("S1", axis=1)
    
    # * Get the Labels * #
    labels = input_df["S1"]
    
    return labels, ftrs


def train_and_test(training_filename: str, test_filename: str):

    # * Read in the Data * #
    train_df = pd.read_csv(training_filename)
    test_df = pd.read_csv(test_filename)

    # * Get the Labels & Features from the Training Data * #
    train_labels, train_ftrs = get_Label_and_Features(train_df)

    # * Create the SVC Model * #
    SYSOUT.write(HDR + 'Creating SVC Model...')
    SVC_model: SVC = SVC(kernel='sigmoid', random_state=SEED)

    # * Train the Model * #
    SVC_model.fit(train_ftrs, train_labels)
    SYSOUT.write(OVERWRITE + ' SVC Model Created '.ljust(50, '-') + SUCCESS)

    # * Create the Dummy Model * #
    # this is module always guesses No Tornado (-1) #
    # dummy_model: DummyClassifier = DummyClassifier(strategy="constant", constant="No Tornado")
    dummy_model: DummyClassifier = DummyClassifier(strategy="constant", constant=-1)

    # * Train the Dummy Model * #
    dummy_model.fit(train_ftrs, train_labels)

    # * Test the Model * #
    SYSOUT.write(HDR + 'Testing SVC Model...\n')
    
    test_labels, test_ftrs = get_Label_and_Features(test_df)

    prediction_dummy = dummy_model.predict(test_ftrs)  # make the dummy prediction
    dummy_score: float = accuracy_score(test_labels, prediction_dummy)  # test the prediction

    prediction = SVC_model.predict(test_ftrs)  # make prediction
    svc_score = accuracy_score(test_labels, prediction)  # test prediction
    
    SYSOUT.write(OVERWRITE + ' SVC Model Tested '.ljust(50, '-') + SUCCESS)

    # ****************************** Report Results ****************************** #
    # * Dummy Model * #
    # Testing Data Report
    print(f'\n{"Dummy Model Report - Testing Report":^59}')
    printStats(test_labels, prediction_dummy, dummy_score, test_df)
    # Training Data Report
    print(f'\n{"Dummy Model Report - Training Report":^59}')
    printStats(train_labels, dummy_model.predict(train_ftrs),
               accuracy_score(train_labels, dummy_model.predict(train_ftrs)), train_df)

    # * SVC Model * #
    # Testing Data Report
    print(f'\n\n\n{"SVC Model Report - Testing Report":^59}')
    printStats(test_labels, prediction, svc_score, test_df)
    # Training Data Report
    print(f'\n{"SVC Model Report - Training Report":^59}')
    printStats(train_labels, SVC_model.predict(train_ftrs),
               accuracy_score(train_labels, SVC_model.predict(train_ftrs)), train_df)


def printStats(true_labels, predicted_labels, score, frame):
    """ Creates a report using various statistics & print the result to the console."""

    frequency = frame['S1'].value_counts().to_frame()

    # Get the True Negatives, False Positives, False Negatives, & True Positives
    TN, FP, FN, TP = confusion_matrix(true_labels, predicted_labels).ravel()
    no_tornados = frequency['S1'][-1]
    tornados = frequency['S1'][1]
    row_0 = f"{f'Tornados = {tornados}':^25} {'||':^4} {f'No Tornados = {no_tornados}':^25}"
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

    size = 57
    print('=' * size)
    print(row_0)
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
    sys.stdout.close()  # close the file stdout is writing to
