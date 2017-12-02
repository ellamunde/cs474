from __future__ import division
from nltk.util import ngrams
import re
#tweet format [(token,tag, polarity)]
def build_feature_vector(tweet,word_count):
    sample=[]

    number_of_words=word_count
    #fractions with respect to number of words in a tweet
    fraction_pos,fraction_neg=get_tokens_fractions(tweet,number_of_words)
    fraction_of_adj,fraction_of_v,fraction_of_n=get_pos_fractions(tweet,number_of_words)
    has_link=len([l for l,tag,pol in tweet if l=='%link'])
    has_mention=len([m for m,tag,pol in tweet if m=='%mention'])
    has_emoticon=len([e for e,tag,pol in tweet if e.startswith('%emoticon')])
    has_number=len([n for n,tag,pol in tweet if n=='%number'])
    #intensifier followed by positives
    intens_followed_by_pos,intens_followed_by_neg,intens_followed_by_neu=check_intens(tweet)
    #negation of positive or negative
    neg_followed_by_pos, neg_followed_by_neg,neg_followed_by_neu = check_negation(tweet)
    future_tense=check_tense(tweet)
    has_modal=check_modal(tweet)
    has_1_person=check_pronoun(tweet)
    contains_pos=containsPos(tweet)
    contains_neg = containsNeg(tweet)
    pos_postion,neg_postion=neg_pos_position(tweet)
    has_excl=len([t for t,tag,pol in tweet if t=="%exclamation"])
    has_stop=len([t for t,tag,pol in tweet if t=="%fullstop"])
    sample=[number_of_words,fraction_pos,fraction_neg,
            fraction_of_adj,fraction_of_n,fraction_of_v,
            has_link,has_mention,has_emoticon,has_number,intens_followed_by_pos,
            intens_followed_by_neg,intens_followed_by_neu,neg_followed_by_pos, neg_followed_by_neg,
            neg_followed_by_neu,future_tense,has_modal,has_1_person,contains_pos,contains_neg,
            pos_postion,neg_postion,has_excl,has_stop]
    return sample


def neg_pos_position(tweet):
    pos=[]
    neg=[]
    for i in range(0,len(tweet)):
        t=tweet[i]
        if t[2]=='pos':
            pos.append(i+1)
        elif t[2]=='neg':
            neg.append(i)
    if len(pos)>0:
        pos=max(pos)
    else:
        pos=0
    if len(neg)>0:
        neg=max(neg)
    else:
        neg=0
    return pos,neg

def check_tense(tweet):
    #in preprocessing replace all 'll with shall
    if len([t for t,tag,pol in tweet if t=='will' or t=='shall']):
        return 1
    else:
        return 0

def containsPos(tweet):
    positives=['like','love']
    if len([t for t, tag, pol in tweet if t in positives]):
        return 1
    else:
        return 0
def containsNeg(tweet):
    negatives=['bad','sad','hate']
    if len([t for t, tag, pol in tweet if t in negatives]):
        return 1
    else:
        return 0
def check_modal(tweet):
    #modals that indicate assumption of events that didn't occur in reality
    #replace 'd with would
    modals=['could','might','may','would']
    if len([t for t, tag, pol in tweet if t in modals]):
        return 1
    else:
        return 0
def check_pronoun(tweet):
    #only 1st person pronoun
    pronouns=['i','me','we','us']
    if len([t for t, tag, pol in tweet if t in pronouns]):
        return 1
    else:
        return 0

def get_tokens_fractions(tweet,word_count):
    num_words=word_count
    pos=len([t for t,tag,pol in tweet if pol=='pos'])
    neg=len([t for t,tag,pol in tweet if pol=='neg'])
    return pos/num_words,neg/num_words

#fration of nouns, adjectives, and verbs
def get_pos_fractions(tweet,word_count):
    num_of_words=word_count
    nouns = len([t for t, tag, pol in tweet if tag == 'n'])
    adj = len([t for t, tag, pol in tweet if tag == 'a'])
    verbs=len([t for t, tag, pol in tweet if tag == 'r'])
    return  nouns/num_of_words,adj/num_of_words,verbs/num_of_words
def check_intens(tweet):
    intensifiers=['so','really','very']
    int_followed_by_pos = 0
    int_followed_by_neg = 0
    int_followed_by_neu=0
    #check for adverbs and intensifiers (in case pos tagger mistake)
    if len([t for t,tag,pol in tweet if tag=='r' or t in intensifiers])>0:
        trigrams=ngrams(tweet,3)
        #format (token,tag,pol)
        for tr in trigrams:
            first=tr[0]
            second=tr[1]
            third=tr[2]
            #if tag
            if first[1]=='r' or first[0] in intensifiers:
                #if polarity
                if second[2]=='pos' or third[2]=='pos':
                    int_followed_by_pos=1
                elif second[2]=='neg' or third[2]=='neg':
                    int_followed_by_neg=1
                elif second[2]=='neu' or third[2]=='neu':
                    int_followed_by_neu=1

    return int_followed_by_pos,int_followed_by_neg,int_followed_by_neu

def check_negation(tweet):
    neg_followed_by_pos = 0
    neg_followed_by_neg = 0
    neg_followed_by_neu = 0
    #check for adverbs and intensifiers (in case pos tagger mistake)
    if len([t for t,tag,pol in tweet if re.match(r'_not',t)])>0:
        trigrams=ngrams(tweet,3)
        #format (token,tag,pol)
        for tr in trigrams:
            first=tr[0]
            second=tr[1]
            third=tr[2]
            #if tag
            if re.match(r'_not',first[0]):
                #if polarity
                if second[2]=='pos' or third[2]=='pos':
                    neg_followed_by_pos=1
                elif second[2]=='neg' or third[2]=='neg':
                    neg_followed_by_neg=1
                elif second[2] == 'neu' or third[2] == 'neu':
                    neg_followed_by_neu = 1
    return neg_followed_by_pos,neg_followed_by_neg,neg_followed_by_neu



