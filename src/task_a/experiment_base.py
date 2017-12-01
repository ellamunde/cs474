import preprocessing
import wordnet
import re
import measurements


def calculate_polarity(tweet_tokens, pos_tokens, neg_tokens):
    sentiment_score = 0

    for t in tweet_tokens:
        # print t
        if re.match(r'^@', t):
            continue

        t = re.sub(r'_', ' ', t)
        # print t
        if t in pos_tokens:
            print ">> positive"
            sentiment_score += 1

        if t in neg_tokens:
            print ">> negative"
            sentiment_score -= 1

    if sentiment_score > 0:
        return 'positive'
    elif sentiment_score < 0:
        return 'negative'
    else:
        return 'neutral'


def test_sentiment(test, final_tokens_pos, final_tokens_neg):
    sentiment_class = []

    for index, values in test.iterrows():
        tokens = preprocessing.extract_tokens(values['CLEANED'])
        tokens = wordnet.add_synonyms(tokens)
        tokens = wordnet.lemmatize_words(tokens)
        # print tokens
        tokens = preprocessing.remove_stopwords(preprocessing.get_tokens_only(tokens))

        sentiment = calculate_polarity(tokens, final_tokens_pos, final_tokens_neg)
        sentiment_class.append(sentiment)
        print "class: " + values['POLARITY']
        print "prediction: " + sentiment

    test["PREDICTION"] = sentiment_class
    return test


train = "train"
test = "test"
pos = "positive"
neg = "negative"
neu = "neutral"

train_a = preprocessing.get_data(train, "A")
# train_a_tokens = preprocessing.get_tokens(train_a, None, 'TWEET')
# print train_a_tokens

# for faster training
# tokens_pos = preprocessing.get_subset(train_a, pos)
# tokens_neg = preprocessing.get_subset(train_a, neg)
# tokens_neu = preprocessing.get_subset(train_a, neu)

# get tokens for all class
tokens_pos = preprocessing.get_tokens(train_a, pos, 'CLEANED')
tokens_neg = preprocessing.get_tokens(train_a, neg, 'CLEANED')
tokens_neu = preprocessing.get_tokens(train_a, neu, 'CLEANED')

# one token only exists in one class
# filtered_pos = preprocessing.filter_tokens(tokens_pos, tokens_neg + tokens_neu)
# filtered_neg = preprocessing.filter_tokens(tokens_neg, tokens_pos + tokens_neu)
# filtered_neu = preprocessing.filter_tokens(tokens_neu, tokens_pos + tokens_neg)
#
# print "pos first filter: "
# print filtered_pos
# print "neg first filter: "
# print filtered_neg
# print "neu first filter: "
# print filtered_neu

# adding synonyms
# syn_pos = wordnet.add_synonyms(filtered_pos)
# syn_neg = wordnet.add_synonyms(filtered_neg)
# syn_neu = wordnet.add_synonyms(filtered_neu)

# adding synonyms
syn_pos = wordnet.add_synonyms(tokens_pos)
syn_neg = wordnet.add_synonyms(tokens_neg)
syn_neu = wordnet.add_synonyms(tokens_neu)

# adding antonyms
ant_syn_pos = wordnet.add_antonyms(syn_pos,syn_neg)
ant_syn_neg = wordnet.add_antonyms(syn_neg,syn_pos)

# filtering (?) (again?)
final_pos = preprocessing.filter_tokens(ant_syn_pos,set(ant_syn_neg)|set(syn_neu))
final_neg = preprocessing.filter_tokens(ant_syn_neg,set(ant_syn_pos)|set(syn_neu))
final_neu = preprocessing.filter_tokens(syn_neu,set(ant_syn_neg)|set(ant_syn_pos))

# print "pos second filter: "
# print final_pos
# print "neg second filter: "
# print final_neg
# print "neu second filter: "
# print final_neu

# tokens only
final_lemma_pos = wordnet.lemmatize_words(final_pos)
final_lemma_neg = wordnet.lemmatize_words(final_neg)
final_lemma_neu = wordnet.lemmatize_words(final_neu)

final_tokens_pos = preprocessing.remove_stopwords(preprocessing.get_tokens_only(final_lemma_pos))
final_tokens_neg = preprocessing.remove_stopwords(preprocessing.get_tokens_only(final_lemma_neg))
final_tokens_neu = preprocessing.remove_stopwords(preprocessing.get_tokens_only(final_lemma_neu))

# print "pos: "
# print final_tokens_pos
# print "neg: "
# print final_tokens_neg
# print "neu: "
# print final_tokens_neu

# testing
test_a = preprocessing.get_data(test, "A")
test_a = test_a[:50]
test_a_result = test_sentiment(test_a, final_tokens_pos, final_tokens_neg)

# test accuracy
accuracy_a = measurements.get_accuracy(test_a_result)
print accuracy_a
