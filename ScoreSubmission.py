import pandas as pd
import numpy as np
import sklearn

SUBMISSION_FILE = r'submission.csv'
LABELS_FILE = r'.csv'

ID_COLUMN = 'id_code'
LABEL_COLUMN = 'diagnosis'

def main():
    df1 = pd.read_csv(SUBMISSION_FILE)
    df2 = pd.read_csv(LABELS_FILE)
    assert len(df1) == len(df2)
    assert df1[ID_COLUMN].equals(df2[LABEL_COLUMN])

    y1 = df1[LABEL_COLUMN].to_numpy()
    y2 = df2[LABEL_COLUMN].to_numpy()

    kappa = sklearn.metrics.cohen_kappa_score(y1, y2, weights='quadratic')
    accur = sklearn.metrics.accuracy_score(y1, y2)
    print("Quadratic Kappa: %.5f Accuracy: %.5f" % (kappa, accur))


if __name__ == '__main__':
    main()
