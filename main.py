"""
main.py creates, trains, and tests machine learning models when provided with test & training data sets.

Authors/Contributors: Dr. Dimitrios Diochnos, Conner Flansburg

Github Repo:
"""
# import logging as log
# import traceback
# import typing as typ
import numpy as np
import typing
from collections import namedtuple
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
    SYSOUT.write(HDR + ' Creating SVC Model...')
    SVC_model: SVC = SVC(C=100, gamma=0.001, kernel='rbf', random_state=SEED)

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
    SYSOUT.write(HDR + ' Testing SVC Model...')
    
    test_labels, test_ftrs = get_Label_and_Features(test_df)
    
    SYSOUT.write(OVERWRITE + ' SVC Model Tested '.ljust(50, '-') + SUCCESS)

    # ****************************** Report Results ****************************** #

    # this named tuple will be used to pass data to the report generator using the results dict
    Report_Data = namedtuple('Report_Data', ['Actual', 'Predicted', 'Frame'])

    # * Dummy Model * #
    d_train = Report_Data(Predicted=dummy_model.predict(train_ftrs), Actual=train_labels, Frame=train_df)
    d_test = Report_Data(Predicted=dummy_model.predict(test_ftrs), Actual=test_labels, Frame=test_df)

    # * SVC Model * #
    SVC_train = Report_Data(Predicted=SVC_model.predict(train_ftrs), Actual=train_labels, Frame=train_df)
    SVC_test = Report_Data(Predicted=SVC_model.predict(test_ftrs), Actual=test_labels, Frame=test_df)

    createReportFrame(d_train, d_test, SVC_train, SVC_test)  # generate the report dataframe


def createReportFrame(d_train, d_test, SVC_train, SVC_test):

    rws = ['Tornados', 'No Tornados', 'True Positives', 'False Positives', 'True Negatives',
           'False Negatives', 'Accuracy', 'Precision', 'Recall/POD', 'F1 Score',
           "Heidke's Score", 'FAR', 'Bias', 'CSI']

    results = {'Tornados': [], 'No Tornados': [], 'True Positives': [], 'False Positives': [],
               'True Negatives': [], 'False Negatives': [], 'Accuracy': [], 'Precision': [],
               'Recall/POD': [], 'F1 Score': [], "Heidke's Score": [], 'FAR': [], 'Bias': [],
               'CSI': []
               }

    # Get stats for each model & dataset
    calculateReport(results, d_train.Actual, d_train.Predicted, d_train.Frame)        # Add d_train to dict
    calculateReport(results, d_test.Actual, d_test.Predicted, d_test.Frame)           # Add d_test to dict
    calculateReport(results, SVC_train.Actual, SVC_train.Predicted, SVC_train.Frame)  # Add SVC_train to dict
    calculateReport(results, SVC_test.Actual, SVC_test.Predicted, SVC_test.Frame)     # Add SVC_test to dict

    if not results.items():  # if the dictionary is empty
        raise Exception('Dictionary Empty!')

    # create the dataframe
    df = pd.DataFrame(results, columns=rws, index=['dummy train', 'dummy test', 'SVC train', 'SVC test'])

    # format the data frame
    df.round(decimals=3)
    df = df.transpose()

    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,):
        print(df)


def calculateReport(results: typing.Dict, true_labels, prediction, frame):

    # *** Get the Frequency of the Two Labels *** #
    frequency = frame['S1'].value_counts().to_frame()
    results['Tornados'].append(frequency['S1'][1])
    results['No Tornados'].append(frequency['S1'][-1])
    # ******************************************* #

    # *** Get the Confusion Matrix & Extract it's Values *** #
    TN, FP, FN, TP = confusion_matrix(true_labels, prediction).ravel()
    results['True Positives'].append(TP)
    results['False Positives'].append(FP)
    results['True Negatives'].append(TN)
    results['False Negatives'].append(FN)
    # ****************************************************** #

    # *** Perform the Calculations & Handle Divide by 0 Case *** #
    # Precision Calculation
    denom = TP + FP
    if denom == 0:  # if we would divide by zero,
        P = 0.000   # then set precision to zero
    else:           # otherwise perform calculation
        P = round(TP / (TP + FP), 3)
    results['Precision'].append(P)

    # Recall Calculation
    denom = TP + FN
    if denom == 0:  # if we would divide by zero,
        R = 0.000   # then set precision to zero
    else:           # otherwise perform calculation
        R = round(TP / (TP + FN), 3)
    results['Recall/POD'].append(R)

    # F1 Score Calculation
    denom = P + R
    if denom == 0:  # if we would divide by zero,
        F1 = 0.000  # then set precision to zero
    else:           # otherwise perform calculation
        F1 = round((2 * ((P * R) / (P + R))), 3)
    results['F1 Score'].append(F1)
    # ****************************************************** #

    # *** Compute the Other Statistics *** #
    results['Accuracy'].append(f'{round((accuracy_score(true_labels, prediction)*100),2)}%')
    results["Heidke's Score"].append(computeHeidkes(TP, TN, FP, FN))
    results['FAR'].append(computeFAR(TP, FP))
    results['Bias'].append(computeBias(TP, FP, FN))
    results['CSI'].append(computeCSI(TP, FP, FN))
    # ************************************ #


def computeHeidkes(tp, tn, fp, fn: int) -> float:
    """ Compute the Heidke's Skill Score. """
    top: int = 2 * (tp*tn - fp*fn)  # compute the numerator
    bottom: int = (tp+fn) * (fn+tn) + (tp+fp) * (fp+tn)  # compute the denominator

    if bottom == 0:
        return np.NaN

    return round(top / bottom, 3)


def computeFAR(tp, fp: int) -> float:
    """ Compute the False Alarm Ratio. """
    bottom: int = tp + fp  # compute the denominator

    if bottom == 0:
        return np.NaN

    return round(fp / bottom, 3)


def computeBias(tp, fp, fn: int) -> float:
    """ Compute the Bias. """
    top: int = tp + fp  # compute the numerator
    bottom: int = tp + fn  # compute the denominator

    if bottom == 0:
        return np.NaN

    return round(top / bottom, 3)


def computeCSI(tp, fp, fn: int) -> float:
    """ Compute the Critical Success Index. """
    bottom: int = tp + fp + fn  # compute the denominator

    if bottom == 0:
        return np.NaN

    return round(tp / bottom, 3)


if __name__ == '__main__':
    main()
    sys.stdout.close()  # close the file stdout is writing to
