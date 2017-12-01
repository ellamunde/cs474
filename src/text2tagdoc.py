from gensim.models.doc2vec import TaggedDocument


class LabeledLineSentence(object):
    def __init__(self, text, label):
        self.text = text
        self.label = label

    def __iter__(self):
        for idx, doc in enumerate(self.text):
            yield TaggedDocument(doc, [self.label[idx]])