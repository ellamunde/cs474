import preprocessing
import wordnet


def calculate_polarity(tweet_tokens, pos_tokens, neg_tokens):
    sentiment_score = 0

    for t in tweet_tokens:
        if t in pos_tokens:
            sentiment_score += 1
        elif t in neg_tokens:
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
        tokens = [t for t, tag in preprocessing.extract_tokens(values['TWEET'])]
        sentiment = calculate_polarity(tokens, final_tokens_pos, final_tokens_neg)
        sentiment_class.append(sentiment)
        print "class: " + values['POLARITY']
        print "prediction: " + sentiment

    test["PREDICTION"] = sentiment_class
    return test


def get_accuracy(result):
    classified_correctly = 0
    for idx, values in result.iterrows():

        if values['POLARITY'] == values['PREDICTION']:
            classified_correctly += 1

    accuracy = classified_correctly / len(result)
    return accuracy


train = "train"
test = "test"
pos = "positive"
neg = "negative"
neu = "neutral"

train_a = preprocessing.get_data(train, "A")
train_a_tokens = preprocessing.get_tokens(train_a, None, 'TWEET')
# print train_a_tokens

# get tokens for all class
tokens_pos = preprocessing.get_tokens(train_a, pos, 'TWEET')
tokens_neg = preprocessing.get_tokens(train_a, neg, 'TWEET')
tokens_neu = preprocessing.get_tokens(train_a, neu, 'TWEET')

# one token only exists in one class
filtered_pos = preprocessing.filter_tokens(tokens_pos, tokens_neg + tokens_neu)
filtered_neg = preprocessing.filter_tokens(tokens_neg, tokens_pos + tokens_neu)
filtered_neu = preprocessing.filter_tokens(tokens_neu, tokens_pos + tokens_neg)

# adding synonyms
syn_pos = wordnet.add_synonyms(filtered_pos)
syn_neg = wordnet.add_synonyms(filtered_neg)
syn_neu = wordnet.add_synonyms(filtered_neu)

# adding antonyms
ant_syn_pos = wordnet.add_antonyms(syn_pos,syn_neg)
ant_syn_neg = wordnet.add_antonyms(syn_neg,syn_pos)

# filtering (?) (again?)
final_pos = preprocessing.filter_tokens(ant_syn_pos,set(ant_syn_neg)|set(syn_neu))
final_neg = preprocessing.filter_tokens(ant_syn_neg,set(ant_syn_pos)|set(syn_neu))
final_neu = preprocessing.filter_tokens(syn_neu,set(ant_syn_neg)|set(ant_syn_pos))

# tokens only
final_tokens_pos = preprocessing.remove_stopwords(preprocessing.get_tokens_only(final_pos))
final_tokens_neg = preprocessing.remove_stopwords(preprocessing.get_tokens_only(final_neg))
final_tokens_neu = preprocessing.remove_stopwords(preprocessing.get_tokens_only(final_neu))

print "pos: "
print final_tokens_pos
print "neg: "
print final_tokens_neg
print "neu: "
print final_tokens_neu

# testing
test_a = preprocessing.get_data(test, "A")
test_a_result = test_sentiment(test_a, final_tokens_pos, final_tokens_neg)

# test accuracy
accuracy_a = get_accuracy(test_a_result)
print accuracy_a
