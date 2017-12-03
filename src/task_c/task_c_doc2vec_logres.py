import pre_task_bc_doc2vec as pre

input_from_file = pre.get_data('train', 'C')[:200]
epoch = 3
model_dm, model_dbow, train_data = pre.get_model(input_from_file, epoch=epoch)
# print train_data
dm_svm_model = pre.logres_polarity_model(model=model_dm,train_data=train_data, multi=True)
dbow_svm_model = pre.logres_polarity_model(model=model_dbow,train_data=train_data, multi=True)

print ">> dm, svm-dm"
dm_prediction = pre.logres_polarity_test(model_dm, dm_svm_model, dataset='C')
print ">> dbow, svm-dbow"
dbow_prediction = pre.logres_polarity_test(model_dbow, dbow_svm_model, dataset='C')
# print ">> dbow, svm-dm"
# dbow_dm_prediction = pre.svm_polarity_test(model_dbow, dm_svm_model, dataset='C')
# print ">> dm, svm-dbow"
# dm_dbow_prediction = pre.svm_polarity_test(model_dm, dbow_svm_model, dataset='C')
