from pandas import DataFrame
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix


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
    for value in true_table.index.tolist():
        true_pos[value] = 0

    # print true_pos
    # process
    for index, row in table.iterrows():
        n_total += 1
        polarity = row['POLARITY']
        prediction = row['PREDICTION']

        if polarity == prediction:
            true_pos[polarity] += 1

    # get recall
    recall = 0
    # print true_pos
    # print type(true_pos)
    for idx, row in true_pos.iteritems():
        # print idx --> label
        # print row --> int
        truepositive = float(row)
        totalpositive = float(true_table[idx])
        recall += truepositive / totalpositive

    average = recall / n_class
    print ">> average recall:"
    print average
    return average


def get_accuracy(table):
    true_positive = 0
    n_total = 0

    for index, row in table.iterrows():
        n_total += 1
        # print row['CLEANED']
        polarity = row['POLARITY']
        prediction = row['PREDICTION']
        # print polarity, prediction

        if polarity == prediction:
            true_positive += 1

    accuracy = true_positive / (n_total * 1.0)
    print ">> accuracy:"
    print accuracy
    return accuracy


def f_pn_measurement(table):
    # F1PN= 1/2 (F1P+F1N)
    # F1P= F1 of positive class
    # F1N= F1 of negative class
    # F = 2PR/(P+R)

    # true value
    true_table = table['POLARITY'].value_counts()

    f_pn = 0
    for idx, row in true_table.iteritems():
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        # print idx
        # print row

        if idx == "neutral":
            continue

        for index in range(len(table)):
            # print index
            point = 0
            polarity = table['POLARITY'][index]
            prediction = table['PREDICTION'][index]
            # print polarity
            # print prediction

            if polarity == prediction:
                if polarity == "positive":
                    tp += 1
                else:
                    tn += 1
            else:
                if prediction == "positive":
                    fp += 1
                else:
                    fn += 1

        precision = float(tp) / float(tp + fp) if float(tp + fp) > 0 else 0
        recall = float(tp) / float(tp + fn) if float(tp + fn) > 0 else 0
        f_measure = 2 * precision * recall / (precision + recall)
        f_pn += f_measure

    f_pn = f_pn / 2
    print ">> f_pn:"
    print f_pn
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
    # print true_table
    for idx, row in true_table.iteritems():
        # number instance with true class C
        # print idx
        # print row
        n_total = row
        data = table[table['POLARITY'] == idx].reset_index(drop=True)
        # print data

        differences = 0
        for data_idx in range(len(data)):
            differences += abs((data['PREDICTION'][data_idx] - data['POLARITY'][data_idx]))

        # print differences
        mae += float(differences) / float(n_total)

    macro_avg = mae / float(n_class)
    print ">> macro avg:"
    print macro_avg
    return macro_avg


def standard_mae(table):
    # MAE(h, Te) = 1 | Tej | xiTej | h(xi) - yi |
    # xi= tweet
    # yi= true label of xi
    # h(xi)= predicted label
    # Tej=test data with true class cj

    differences = 0
    n_total = 0
    # print table
    for idx in range(len(table)):
        # print idx
        n_total += 1
        differences += abs((table['PREDICTION'][idx] - table['POLARITY'][idx]))

    mae = float(differences) / float(n_total)
    print ">> mae:"
    print mae
    return mae


def predict(text_test, label_test, model):
    prediction = model.predict(text_test)
    print ">> model score: "
    print model.score(text_test, label_test)
    print ">> model report: "
    pol_pre = label_test.to_frame().reset_index(drop=True).join(DataFrame({'PREDICTION': prediction}))
    print pol_pre
    print classification_report(label_test, prediction)
    return pol_pre