#!/usr/bin/env python

import numpy as np  # for np.mean() and np.std()
import nltk, sys, inspect
import nltk.corpus.util
from nltk.corpus import inaugural, brown, conll2007  # import corpora
from nltk.corpus import stopwords  # stopword list

# Import the Twitter corpus and LgramModel
try:
    from nltk_model import *  # See the README inside the nltk_model folder for more information
    from twitter import *
except ImportError:
    from .nltk_model import *  # Compatibility depending on how this script was run
    from .twitter import *

twitter_file_ids = xtwc.fileids()

#################### SECTION A: COMPARING CORPORA ####################

##### Solution for question 1 #####

def get_corpus_tokens(corpus, list_of_files):
    '''Get the raw word tokens from (part of) an NLTK corpus

    :type corpus: nltk.corpus.CorpusReader
    :param corpus: An NLTK corpus
    :type list_of_files: list(file)
    :param list_of_files: files to read from
    :rtype: list(str)
    :return: the tokenised contents of the files'''

    # Return the list of corpus tokens
    tokens = []
    for file in list_of_files:
        tokens += list(corpus.words(file))
    return tokens

def q1(corpus, list_of_files):
    '''Compute the average word type length from (part of) a corpus

    :type corpus: nltk.corpus.CorpusReader
    :param corpus: An NLTK corpus
    :type list_of_files: list(str)
    :param list_of_files: names of files to read from
    :rtype: float
    :return: the average word type length over all the files'''

    # raise NotImplementedError # remove when you finish defining this function
    # Get a list of all tokens in the corpus
    corpus_tokens = get_corpus_tokens(corpus, list_of_files)

    # Construct a list that contains the token lengths for each DISTINCT token in the document
    distinct_token_lengths = [len(w) for w in set(v.lower() for v in corpus_tokens)]

    # Return the average distinct token (== type) length
    return float(sum(distinct_token_lengths)) / len(distinct_token_lengths)

##### Solution for question 2 #####

def q2():
    '''Question: Why might the average type length be greater for
       twitter data?

    :rtype: str
    :return: your answer [max 500 chars]'''

    return inspect.cleandoc("""\
    The average word type length for the Twitter data is greater because
    it contains long tokens such as a website (e.g. 'http://bit.ly/d17rdJ'),
    a user ID (e.g. '@user10414528') and even a sentence in other language
    (e.g. '朝目がさめて真っ先にラマーズPの新曲' in Japanese) which does not use
    spaces to separate words.""")[0:500]

#################### SECTION B: DATA IN THE REAL WORLD ####################

##### Solution for question 3 #####

def q3(corpus, list_of_files, x):
    '''Tabulate and plot the top x most frequently used word types
       and their counts from the specified files in the corpus

    :type corpus: nltk.corpus.CorpusReader
    :param corpus: An NLTK corpus
    :type list_of_files: list(str)
    :param list_of_files: names of files to read from
    :rtype: list(tuple(string,int))
    :return: top x word types and their counts from the files'''

    # Get a list of all tokens in the corpus
    corpus_tokens = get_corpus_tokens(corpus, list_of_files)

    # Construct a frequency distribution over the lowercased types in the document
    fd_doc_types = nltk.FreqDist(w.lower() for w in corpus_tokens)

    # Find the top x most frequently used types in the document
    top_types = fd_doc_types.most_common(x)
    # Produce a plot showing the top x types and their frequencies
    fd_doc_types.plot(x)

    # Return the top x most frequently used types
    return top_types

##### Solution for question 4 #####

def q4(corpus_tokens):
    '''Clean a list of corpus tokens

    :type corpus_tokens: list(str)
    :param corpus_tokens: raw corpus tokens
    :rtype: list(str)
    :return: cleaned, lower-cased list of corpus tokens'''

    stops = list(stopwords.words("english"))

    # A token is 'clean' if it's alphanumeric and NOT in the list of stopwords
    clean = []
    for token in corpus_tokens:
        if token.isalnum():
            token = token.lower()
            if token not in stops: # alphanumeric & not stop
                clean.append(token)

    # Return a cleaned list of corpus tokens
    return clean

##### Solution for question 5 #####

def q5(cleaned_corpus_tokens, x):
    '''Tabulate and plot the top x most frequently used word types
       and their counts from the corpus tokens

    :type corpus_tokens: list(str)
    :param corpus_tokens: (cleaned) corpus tokens
    :rtype: list(tuple(string,int))
    :return: top x word types and their counts from the files'''

    # Construct a frequency distribution over the lowercased tokens in the document
    fd_doc_types = nltk.FreqDist([w.lower() for w in cleaned_corpus_tokens])

    # Find the top x most frequently used types in the document
    top_types = fd_doc_types.most_common(x)

    # Produce a plot showing the top x types and their frequencies
    fd_doc_types.plot(x)

    # Return the top x most frequently used types
    return top_types

##### Solution for question 6 #####

def q6():
    '''Problem: noise in Twitter data

    :rtype: str
    :return: your answer [max 1000 chars]'''

    return inspect.cleandoc("""\
    The original Twitter data contain non-alphanumeric tokens such as
    a website, a user ID started with @ sign. Also, the language used
    is not purely English. Japanese tokens occur frequently but they are not
    tokenized properly because Japanese words are not separated by spaces.
    In addition, there are plenty of slang, abbreviation, and typos, all of
    which will cause significant noise. The implemented cleaning has removed
    a lot of noisy data such as the websites, user IDs, English stopwords,
    punctuations and so on. My suggestion for further cleaning is to divide
    raw tokens into different language categories, and use the language-specific
    tokenizer to process the data separately, e.g. using jNlp.jTokenize to
    process Japanese data. Another thing that can be done is to remove common words
    (e.g. the) and rare words because the former has limited information
    whereas the latter is dominated by noise. Spelling correction could also be
    an option (dealing with typos).
    """)[0:1000]

#################### SECTION C: LANGUAGE IDENTIFICATION ####################

##### Solution for question 7 #####

def q7(corpus):
    '''Build a bigram letter language model using LgramModel
       based on the all-alpha subset of the entire corpus

    :type corpus: nltk.corpus.CorpusReader
    :param corpus: An NLTK corpus
    :rtype: LgramModel
    :return: A padded letter bigram model based on nltk.model.NgramModel'''

    # subset the corpus to only include all-alpha tokens
    corpus_tokens = [w.lower() for w in corpus.words() if w.isalpha()]

    # return a smoothed padded bigram letter language model
    lm = LgramModel(2, corpus_tokens, pad_left=True, pad_right=True)

    return lm

##### Solution for question 8 #####

def q8(file_name,bigram_model):
    '''Using a character bigram model, compute sentence entropies
       for a subset of the tweet corpus, removing all non-alpha tokens and
       tweets with less than 5 all-alpha tokens

    :type file_name: str
    :param file_name: twitter file to process
    :rtype: list(tuple(float,list(str)))
    :return: ordered list of average entropies and tweets'''

    list_of_tweets = xtwc.sents(file_name)

    # Remove non-alpha tokens
    list_of_tweets_alpha = [
        [ w.lower() for w in tweet if w.isalpha()]
        for tweet in list_of_tweets
    ]

    # Rmove tweets with < 5 tokens
    cleaned_list_of_tweets = [
        tweet for tweet in list_of_tweets_alpha
        if len(tweet) >= 5
    ]

    # Construct a list of tuples of the form: (entropy,tweet)
    #  for each tweet in the cleaned corpus, where entropy is the
    #  average word for the tweet, and return the list of
    #  (entropy,tweet) tuples sorted by entropy

    entropy_tweet_list = []
    for tweet in cleaned_list_of_tweets:
        avg_word_cross_entropy = np.mean([
            bigram_model.entropy(
                word, pad_left=True,
                pad_right=True, perItem=True
            ) for word in tweet
        ])
        entropy_tweet_list.append((avg_word_cross_entropy, tweet))

    return sorted(entropy_tweet_list, key=lambda t: t[0])

##### Solution for question 9 #####

def q9():
    '''Question: What differentiates the beginning and end of the list
       of tweets and their entropies?

    :rtype: str
    :return: your answer [max 500 chars]'''

    return inspect.cleandoc("""\
    Observe that tweets formed by common English words (e.g. the, and)
    are in the top list, whereas tweets containing long Japanese tokens
    are in the bottom list. Since our bigram model is based on the Brown
    corpus which contains English words only, it will treat characters in
    other language as unseen data, thus the model becomes very "uncertain"
    when predicting the next character, resulting in a higher cross-entropy.
    """)[0:500]

##### Solution for question 10 #####

def q10(list_of_tweets_and_entropies):
    '''Compute entropy mean, standard deviation and using them,
       likely non-English tweets in the all-ascii subset of list of tweets
       and their biletter entropies

    :type list_of_tweets_and_entropies: list(tuple(float,list(str)))
    :param list_of_tweets_and_entropies: tweets and their
                                    english (brown) average biletter entropy
    :rtype: tuple(float, float, list(tuple(float,list(str)))
    :return: mean, standard deviation, ascii tweets and entropies,
             not-English tweets and entropies'''

    # Find the "ascii" tweets - those in the lowest-entropy 90%
    #  of list_of_tweets_and_entropies
    threshold = 0.9
    up_to = int(threshold * len(list_of_tweets_and_entropies))
    list_of_ascii_tweets_and_entropies = list_of_tweets_and_entropies[ :up_to]

    # Extract a list of just the entropy values
    list_of_entropies = [t[0] for t in list_of_ascii_tweets_and_entropies]

    # Compute the mean of entropy values for "ascii" tweets
    mean = np.mean(list_of_entropies)

    # Compute their standard deviation
    standard_deviation = np.std(list_of_entropies)

    # Get a list of "probably not English" tweets, that is, "ascii"
    # tweets with an entropy greater than (mean + (0.674 * std_dev))

    threshold = mean + (0.674 * standard_deviation)
    list_of_not_English_tweets_and_entropies = [t for t in list_of_ascii_tweets_and_entropies if t[0] > threshold]
    list_of_not_English_tweets_and_entropies = sorted(list_of_not_English_tweets_and_entropies, key=lambda t: t[0])

    # Return mean, stdev,
    #  list_of_ascii_tweets_and_entropies,
    #  list_of_not_English_tweets_and_entropies
    return [mean,
            standard_deviation,
            list_of_ascii_tweets_and_entropies,
            list_of_not_English_tweets_and_entropies]

##### Solution for question 11 #####

def q11(list_of_files, list_of_not_English_tweets_and_entropies):
    '''Build a padded spanish bigram letter bigram model and use it
       to re-sort the probably-not-English data

    :type list_of_files: list(str)
    :param list_of_files: spanish corpus files
    :type list_of_tweets_and_entropies: list(tuple(float,list(str)))
    :param list_of_tweets_and_entropies: tweets and their
                                    english (brown) average biletter entropy
    :rtype: list(tuple(float,list(str)))
    :return: probably-not-English tweets and _spanish_ entropies'''

    # Build a bigram letter language model using "LgramModel"
    #  and lower-cased alphanumeric spanish tokens
    spanish_tokens = [w.lower() for w in
        get_corpus_tokens(conll2007, list_of_files) if w.isalpha()]

    # return a smoothed padded bigram letter language model
    lm = LgramModel(2, spanish_tokens, pad_left=True, pad_right=True)

    # Return a list of the input tweets with their english entropy
    # replaced with their entropy using the new bigram letter spanish
    # language model, sorted
    tweets = [t[1] for t in list_of_not_English_tweets_and_entropies]

    entropy_tweet_list = []
    for tweet in tweets:
        avg_word_cross_entropy = np.mean([
            lm.entropy(
                word, pad_left=True,
                pad_right=True, perItem=True
            ) for word in tweet
        ])
        entropy_tweet_list.append((avg_word_cross_entropy, tweet))

    return sorted(entropy_tweet_list, key=lambda t: t[0])

##### Answers #####

def ppEandT(eAndTs):
    '''Pretty print a list of entropy+tweet pairs

    :type eAndTs: list(tuple(float,list(str)))
    :param eAndTs: entropies and tweets
    :return: None'''

    for entropy,tweet in eAndTs:
        print("%.3f {%s}"%(entropy,", ".join(tweet)))

def answers():
    # So we can see these during development
    global answer1a, answer1b, answer2, answer3a, answer3b, answer4a, answer4b
    global answer5a, answer5b, answer6, brown_bigram_model, answer8, answer9
    global answer10, answer11
    ### Question 1
    print("*** Question 1 ***")
    answer1a = q1(inaugural,inaugural.fileids())
    print("Average token length for inaugural corpus: %.2f"%answer1a)
    answer1b = q1(xtwc,twitter_file_ids)
    print("Average token length for twitter corpus: %.2f"%answer1b)
    ### Question 2
    print("*** Question 2 ***")
    answer2 = q2()
    print(answer2)
    ### Question 3
    print("*** Question 3 ***")
    print("Most common 50 types for the inaugural corpus:")
    answer3a = q3(inaugural,inaugural.fileids(),50)
    print(answer3a)
    print("Most common 50 types for the twitter corpus:")
    answer3b = q3(xtwc,twitter_file_ids,50)
    print(answer3b)
    ### Question 4
    print("*** Question 4 ***")
    corpus_tokens = get_corpus_tokens(inaugural,inaugural.fileids())
    answer4a = q4(corpus_tokens)
    print("Inaugural Speeches:")
    print("Number of tokens in original corpus: %s"%len(corpus_tokens))
    print("Number of tokens in cleaned corpus: %s"%len(answer4a))
    print("First 100 tokens in cleaned corpus:")
    print(answer4a[:100])
    print("-----")
    corpus_tokens = get_corpus_tokens(xtwc,twitter_file_ids)
    answer4b = q4(corpus_tokens)
    print("Twitter:")
    print("Number of tokens in original corpus: %s"%len(corpus_tokens))
    print("Number of tokens in cleaned corpus: %s"%len(answer4b))
    print("First 100 tokens in cleaned corpus:")
    print(answer4b[:100])
    ### Question 5
    print("*** Question 5 ***")
    print("Most common 50 types for the cleaned inaugural corpus:")
    answer5a = q5(answer4a, 50)
    print(answer5a)
    print("Most common 50 types for the cleaned twitter corpus:")
    answer5b = q5(answer4b, 50)
    print(answer5b)
    ### Question 6
    print("*** Question 6 ***")
    answer6 = q6()
    print(answer6)
    ### Question 7
    print("*** Question 7: building brown bigram letter model ***")
    brown_bigram_model = q7(brown)
    ### Question 8
    print("*** Question 8 ***")
    answer8 = q8("20100128.txt",brown_bigram_model)
    print("Best 10 english entropies:")
    ppEandT(answer8[:10])
    print("Worst 10 english entropies:")
    ppEandT(answer8[-10:])
    ### Question 9
    print("*** Question 9 ***")
    answer9 = q9()
    print(answer9)
    ### Question 10
    print("*** Question 10 ***")
    answer10 = q10(answer8)
    print("Mean: %s"%answer10[0])
    print("Standard Deviation: %s"%answer10[1])
    print("'Ascii' tweets: Best 10 english entropies:")
    ppEandT(answer10[2][:10])
    print("'Ascii' tweets: Worst 10 english entropies:")
    ppEandT(answer10[2][-10:])
    print("Probably not English tweets: Best 10 english entropies:")
    ppEandT(answer10[3][:10])
    print("Probably not English tweets: Worst 10 english entropies:")
    ppEandT(answer10[3][-10:])
    ### Question 11
    print("*** Question 11 ***")
    list_of_not_English_tweets_and_entropies = answer10[3]
    answer11 = q11(["esp.test","esp.train"],list_of_not_English_tweets_and_entropies)
    print("Probably not English tweets: Best 10 spanish entropies:")
    ppEandT(answer11[:10])
    print("Probably not English tweets: Worst 10 spanish entropies:")
    ppEandT(answer11[-10:])

if __name__ == '__main__':
    answers()
