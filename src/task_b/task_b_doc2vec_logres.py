import src.pre_task_bc_doc2vec as pre

dataset = 'B'
input_from_file = pre.get_data('train', dataset)[:50]
test_set = pre.get_data('test', dataset)[:100]

epoch = 3
model_dm, model_dbow, train_data = pre.get_model(input_from_file, epoch=epoch)

dm_svm_model = pre.polarity_model(d2v_model=model_dm,model='logres', train_data=train_data, multi=False)
dbow_svm_model = pre.polarity_model(d2v_model=model_dbow,model='logres', train_data=train_data, multi=False)

print ">> dm, svm-dm"
dm_prediction = pre.polarity_test(model_dm, dm_svm_model, test_set)
print ">> dbow, svm-dbow"
dbow_prediction = pre.polarity_test(model_dbow, dbow_svm_model, test_set)


import src.measurements as m
m.get_accuracy(dm_prediction)
m.avg_recall(dm_prediction)
m.get_accuracy(dbow_prediction)
m.avg_recall(dbow_prediction)
# print ">> dbow, svm-dm"
# dbow_dm_prediction = pre.svm_polarity_test(model_dbow, dm_svm_model, dataset='B')
# print ">> dm, svm-dbow"
# dm_dbow_prediction = pre.svm_polarity_test(model_dm, dbow_svm_model, dataset='B')
