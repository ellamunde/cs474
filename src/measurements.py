# F1PN= 12 (F1P+F1N)
# F1P= F1 of positive class
# F1N= F1 of negative class

# table:
# name, class, prediction


def avg_recall(table):
    # AvgRec = 1N (R1+ ... + RN)
    # N = number of sentiment class
    # Ri = Recall for each class

    true_pos = {}

    # true value
    true_table = table['CLASS'].value_counts()

    # initialization
    for index, row in true_table.iterrows():
        true_pos[index] = 0

    # process
    for index, row in table.iterrows():
        name = row['CNAME']
        polarity = row['CLASS']
        prediction = row['PREDICTION']

        if polarity == prediction:
            true_pos['CLASS'] += 1
    pass