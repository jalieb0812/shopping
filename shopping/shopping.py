import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):

    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)

        evidence = []

        labels = []
        for row in reader:

            new_list = []


            new_list.append(int(row['Administrative']))

            new_list.append(float(row['Administrative_Duration']))

            new_list.append(int(row['Informational']))

            new_list.append(float(row['Informational_Duration']))

            new_list.append(int(row['ProductRelated']))

            new_list.append(float(row['ProductRelated_Duration']))

            new_list.append(float(row['BounceRates']))

            new_list.append(float(row['ExitRates']))

            new_list.append(float(row['PageValues']))

            new_list.append(float(row['SpecialDay']))

            month_int = 0

            if row['Month'] == 'January':
                month_int = 0

            if row['Month'] == 'Feb':
                month_int = 1

            if row['Month'] == 'Mar':
                month_int = 2

            if row['Month'] == 'April':
                month_int = 3

            if row['Month'] == 'May':
                month_int = 4

            if row['Month'] == 'June':
                month_int = 5

            if row['Month'] == 'Jul':
                month_int = 6

            if row['Month'] == 'Aug':
                month_int = 7

            if row['Month'] == 'Sep':
                month_int = 8

            if row['Month'] == 'Oct':
                month_int = 9

            if row['Month'] == 'Nov':
                month_int = 10

            if row['Month'] == 'Dec':
                month_int = 11

            new_list.append(int(month_int))

            new_list.append(int(row['OperatingSystems']))

            new_list.append(int(row['Browser']))

            new_list.append(int(row['Region']))

            new_list.append(int(row['TrafficType']))

            if row['VisitorType'] == 'Returning_Visitor':
                new_list.append(1)

            if row['VisitorType'] == 'New_Visitor':
                new_list.append(0)


            if row['Weekend'] == "FALSE":
                new_list.append(0)

            if row['Weekend'] == "TRUE":
                new_list.append(1)

            evidence.append(new_list)


            if row['Revenue'] == 'FALSE':
                labels.append(0)

            if row['Revenue'] == 'TRUE':
                labels.append(1)






            #evidence.append(row[0:17])
            #labels.append(row[17])
            #print(row['Administrative'])



        #del evidence[0]
        #del labels[0]

        #evidence_length = len(evidence)

        #labels_length = len(labels)

        #print(f"evidence_length: {evidence_length} \n")

        #print(f"labels_length: {labels_length} \n")

        print(f"labels: {labels} \n ")

        print(f"evidence: {evidence[0]} \n")

    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    return (evidence, labels)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    raise NotImplementedError


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
