import preprocessing
import os

# txt = "Won the match #getin . Plus\u002c tomorrow is a very busy day\u002c with Awareness Day\u2019s and debates. Gulp. Debates..."
train = "train"
test = "test"
train_a = preprocessing.get_data(train, "A")
print train_a
