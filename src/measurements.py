def avg_recall(table):
    # AvgRec = 1/N (R1+ ... + RN)
    # N = number of sentiment class
    # Ri = Recall for each class

    true_pos = {}
    n_total = 0

    # true value
    true_table = table['CLASS'].value_counts()
    n_class = len(true_table)

    # initialization
    for idx, row in true_table.iterrows():
        true_pos[idx] = 0

    # process
    for index, row in table.iterrows():
        n_total += 1
        polarity = row['CLASS']
        prediction = row['PREDICTION']

        if polarity == prediction:
            true_pos['CLASS'] += 1

    # get recall
    recall = 0
    for idx, row in true_pos.iteritems():
        truepositive = float(row[idx])
        totalpositive = float(true_table[idx])
        recall += truepositive / totalpositive

    average = recall / n_class
    return average


def get_accuracy(table):
    true_positive = 0
    n_total = 0

    for index, row in table.iterrows():
        n_total += 1
        polarity = row['CLASS']
        prediction = row['PREDICTION']

        if polarity == prediction:
            true_positive += 1

    accuracy = true_positive / n_total
    return accuracy


def f_pn_measurement(table):
    # F1PN= 1/2 (F1P+F1N)
    # F1P= F1 of positive class
    # F1N= F1 of negative class
    # F = 2PR/(P+R)

    true_pos = {}
    n_total = 0

    # true value
    true_table = table['CLASS'].value_counts()
    n_class = len(true_table)

    f_pn = 0
    for idx, row in true_table.iterrows():
        tp = 0
        fp = 0
        tn = 0
        fn = 0

        if idx == "neutral":
            continue

        for index, instance in table.iterrows():
            point = 0
            polarity = instance['CLASS']
            prediction = instance['PREDICTION']

            if polarity == prediction:
                if polarity == idx:
                    tp += 1
                else:
                    tn += 1
            else:
                if prediction == idx:
                    fp += 1
                else:
                    fn += 1

        precision = float(tp) / float(tp + fp)
        recall = float(tp) / float(tp + fn)
        f_measure = 2 * precision * recall / (precision + recall)
        f_pn += f_measure

    f_pn = f_pn / 2
    return f_pn
