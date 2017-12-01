def avg_recall(table):
    # AvgRec = 1/N (R1+ ... + RN)
    # N = number of sentiment class
    # Ri = Recall for each class

    true_pos = {}
    n_total = 0

    # true value
    true_table = table['POLARITY'].value_counts()
    n_class = len(true_table)

    # initialization
    for idx, row in true_table.iterrows():
        true_pos[idx] = 0

    # process
    for index, row in table.iterrows():
        n_total += 1
        polarity = row['POLARITY']
        prediction = row['PREDICTION']

        if polarity == prediction:
            true_pos['POLARITY'] += 1

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
        print row['CLEANED']
        polarity = row['POLARITY']
        prediction = row['PREDICTION']
        print polarity, prediction

        if polarity == prediction:
            true_positive += 1

    accuracy = true_positive / n_total
    return accuracy


def f_pn_measurement(table):
    # F1PN= 1/2 (F1P+F1N)
    # F1P= F1 of positive class
    # F1N= F1 of negative class
    # F = 2PR/(P+R)

    # true value
    true_table = table['POLARITY'].value_counts()

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
            polarity = instance['POLARITY']
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


def macro_average_mae(table):
    # MAEM(h, Te)=1|C|j=1C1|Tej|xiTej|h(xi)-yi|
    # xi= tweet
    # yi= true label of xi
    # h(xi)= predicted label
    # Tej=test data with true class cj
    # C= ordinal class
    true_pos = {}
    n_total = 0

    # true value
    true_table = table['POLARITY'].value_counts()
    n_class = len(true_table)

    mae = 0
    for idx, row in true_table.iterrows():
        # number instance with true class C
        n_total = row[idx]
        data = table[table['POLARITY'] == idx]

        differences = 0
        for data_idx, data_row in data.itterrows():
            differences += abs((data_row['PREDICTION'] - data_row['POLARITY']))

        mae += float(differences) / float(n_total)

    macro_avg = mae / float(n_class)
    return macro_avg


def standard_mae(table):
    # MAE(h, Te) = 1 | Tej | xiTej | h(xi) - yi |
    # xi= tweet
    # yi= true label of xi
    # h(xi)= predicted label
    # Tej=test data with true class cj

    differences = 0
    n_total = 0
    for idx, row in table.itterrows():
        n_total += 1
        differences += abs((row['PREDICTION'] - row['POLARITY']))

    mae = float(differences) / float(n_total)
    return mae
