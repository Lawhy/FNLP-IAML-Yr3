#!/usr/bin/env python

import nltk, inspect

from nltk.corpus import brown
from nltk.tag import map_tag

assert map_tag('brown', 'universal', 'NR-TL') == 'NOUN', '''
Brown-to-Universal POS tag map is out of date.'''

# module for computing a Conditional Frequency Distribution
from nltk.probability import ConditionalFreqDist

# module for computing a Conditional Probability Distribution
from nltk.probability import ConditionalProbDist

import numpy as np

class HMM:
    def __init__(self, train_data, test_data):
        """
        Initialise a new instance of the HMM.

        :param train_data: The training dataset, a list of sentences with tags
        :type train_data: list(list(tuple(str,str)))
        :param test_data: the test/evaluation dataset, a list of sentence with tags
        :type test_data: list(list(tuple(str,str)))
        """
        self.train_data = train_data
        self.test_data = test_data

        # Emission and transition probability distributions
        self.emission_PD: ConditionalProbDist = None
        self.transition_PD: ConditionalProbDist = None
        self.states = []

        self.viterbi = []
        self.backpointer = []

    # Compute emission model using ConditionalProbDist with the estimator:
    # Lidstone probability distribution with +0.01 added to the sample count for each bin and an extra bin
    def emission_model(self, train_data):
        """
        Compute an emission model using a ConditionalProbDist.

        :param train_data: The training dataset, a list of sentences with tags
        :type train_data: list(list(tuple(str,str)))
        :return: The emission probability distribution and a list of the states
        :rtype: Tuple[ConditionalProbDist, list(str)]
        """

        # TODO prepare data
        data = []
        # Don't forget to lowercase the observation otherwise it mismatches the test data
        for sent in train_data:
            data += [ (tag, word.lower()) for (word, tag) in sent]

        # TODO compute the emission model
        emission_FD = ConditionalFreqDist(data)
        estimator = lambda f: nltk.LidstoneProbDist(f, 0.01, f.B()+1)
        self.emission_PD = ConditionalProbDist(emission_FD, estimator)
        self.states = [s for s in self.emission_PD.keys()]

        return self.emission_PD, self.states

    # Compute transition model using ConditionalProbDist with the estimator:
    # Lidstone probability distribution with +0.01 added to the sample count for each bin and an extra bin
    def transition_model(self, train_data):
        """
        Compute an transition model using a ConditionalProbDist.

        :param train_data: The training dataset, a list of sentences with tags
        :type train_data: list(list(tuple(str,str)))
        :return: The transition probability distribution
        :rtype: ConditionalProbDist
        """
        # TODO: prepare the data
        data = []

        # The data object should be an array of tuples of conditions and observations,
        # in our case the tuples will be of the form (tag_(i),tag_(i+1)).
        # DON'T FORGET TO ADD THE START SYMBOL <s> and the END SYMBOL </s>
        for s in train_data:
            data.append(("<s>", s[0][1]))
            for i in range(len(s) - 1):
                data.append((s[i][1], s[i+1][1]))
            data.append((s[len(s) - 1][1], "</s>"))

        # TODO compute the transition model
        transition_FD = ConditionalFreqDist(data)
        estimator = lambda f: nltk.LidstoneProbDist(f, 0.01, f.B() + 1)
        self.transition_PD = ConditionalProbDist(transition_FD, estimator)

        return self.transition_PD

    # Train the HMM
    def train(self):
        """
        Trains the HMM from the training data
        """
        self.emission_model(self.train_data)
        self.transition_model(self.train_data)

    # Part B: Implementing the Viterbi algorithm.

    # Initialise data structures for tagging a new sentence.
    # Describe the data structures with comments.
    # Use the models stored in the variables: self.emission_PD and self.transition_PD
    # Input: first word in the sentence to tag
    def initialise(self, observation):
        """
        Initialise data structures for tagging a new sentence.

        :param observation: the first word in the sentence to tag
        :type observation: str
        """
        # Initialise viterbi, including
        #  transition from <s> to observation
        # use costs (-log-base-2 probabilities)
        # TODO
        # empty everything
        self.viterbi = dict()
        self.backpointer = dict()
        # lambda expression of the sum of negative log probs
        cost = lambda p, q: - float(p + q)
        # The Viterbi table should be m*n where m is the number of states
        # and n is the number of words.
        # Initialliy, for each state, we calculate the emission probability
        # (the prob of observation given the state), and the transition
        # probability (state given the start symbol), sum the negative logs of
        # them to get the corresponding cost.
        # I chose to use dict() to implement the Viterbi table because it supports
        # a pair of keys, i.e. [state, t]
        for i in range(len(self.states)):
            state = self.states[i]
            p_obs_given_pos = self.emission_PD[state].logprob(observation)
            p_pos_given_start = self.transition_PD['<s>'].logprob(state)
            self.viterbi[state, 0] = cost(p_obs_given_pos, p_pos_given_start)

            # Initialise backpointer
            # TODO
            # Initialise the backpointer by filling in m 0s. Again, use the pair
            # key: [state, t].
            self.backpointer[state, 0] = 0



    # Tag a new sentence using the trained model and already initialised data structures.
    # Use the models stored in the variables: self.emission_PD and self.transition_PD.
    # Update the self.viterbi and self.backpointer datastructures.
    # Describe your implementation with comments.
    # Input: list of words
    def tag(self, observations):
        """
        Tag a new sentence using the trained model and already initialised data structures.

        :param observations: List of words (a sentence) to be tagged
        :type observations: list(str)
        :return: List of tags corresponding to each word of the input
        """
        tags = []

        # lambda expression of the sum of negative log probs
        cost = lambda p, q: - float(p + q)

        for t in range(1, len(observations)):
            # the new observation
            observation = observations[t]
            for i in range(len(self.states)):
                state = self.states[i]
                # TODO update the viterbi and backpointer data structures
                # compute the emission prob
                p_emission = self.emission_PD[state].logprob(observation)
                # compute all the possible costs
                costs = []
                for j in range(len(self.states)):
                    pre_state = self.states[j]
                    pre_cost = self.viterbi[pre_state, t-1]
                    p_transition = self.transition_PD[pre_state].logprob(state)
                    # new cost should be sum of the previous cost (corresponding
                    # to the jth state) and the newly computed cost
                    new_cost = cost(p_emission, p_transition) + pre_cost
                    costs.append(new_cost)
                # pick out the minimum cost
                min_cost_index = np.argmin(costs)
                # update the Viterbi table with the best cost
                self.viterbi[state, t] = costs[min_cost_index]
                # update the backpointer with the best index
                self.backpointer[state, t] = min_cost_index

        # TODO
        # Add cost of termination step (for transition to </s> , end of sentence).
        terminal_costs = []
        # for each of the final states, compute the cost which consists of the previous
        # cost that corresponds to the state, and the transition probability only (the
        # emission probability does not matter because there is no emission at all!).
        for s in range(len(self.states)):
            last_state = self.states[s]
            last_cost = self.viterbi[last_state, len(observations) - 1]
            last_p_transition = self.transition_PD[last_state].logprob('</s>')
            terminal_cost = cost(0, last_p_transition) + last_cost
            terminal_costs.append(terminal_cost)
        # find out the best path by selecting the path that gives the lowest overall cost
        best_path_cost = min(terminal_costs)
        best_path_index = np.argmin(terminal_costs)
        # complete the viterbi and backpointer tables
        self.viterbi['</s>', len(observations)] = best_path_cost
        self.backpointer['</s>', len(observations)] = best_path_index

        # TODO
        # Reconstruct the tag sequence using the backpointer list.
        # Return the tag sequence corresponding to the best path as a list.
        # The order should match that of the words in the sentence.

        # backtrack the best states chosen by the algorithm
        best_path = []
        choice = '</s>'
        for t in range(len(observations), 0, -1):
            index = self.backpointer[choice, t]
            choice = self.states[index]
            best_path.append(choice)
        # reverse the backtrace to obtain the desired tags
        tags = reversed(best_path)

        return tags

def answer_question4b():
    """ Report a tagged sequence that is incorrect
    :rtype: str
    :return: your answer [max 280 chars]"""

    tagged_sequence = tagged_sent
    correct_sequence = correct_sent
    # Why do you think the tagger tagged this example incorrectly?
    answer =  inspect.cleandoc("""\
    The transition probs are generally larger than the emission probs
    (thus dominate in cost computation). Thus, the HMM tagger performs
     badly on compound nouns (e.g. City Executive Committee). In our
    example, 'ADJ' after 'NOUN' is rare, so 'Executive' is tagged as 'NOUN'
    wrongly.
    """)[0:280]

    return tagged_sequence, correct_sequence, answer

def answer_question5():
    """Suppose you have a hand-crafted grammar that has 100% coverage on
        constructions but less than 100% lexical coverage.
        How could you use a POS tagger to ensure that the grammar
        produces a parse for any well-formed sentence,
        even when it doesn't recognise the words within that sentence?

    :rtype: str
    :return: your answer [max 500 chars]"""

    return inspect.cleandoc("""\
    The pre-trained POS tagger can be used to tag words that are not covered
    by the hand-crafted lexicon. In this way, any input sentence will be fully
    tagged and thus the hand-crafted grammar can parse it (e.g. Suppose originally
    we have a simple sentence with tags 'NOUN', 'unknown', 'NOUN', the POS tagger
    can fill the gap by replacing 'unknown' with 'VERB'). This approach is
    expected to be do better because the original parser cannot parse sentences
    with missing tags.""")[0:500]

# Useful for testing
def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    # http://stackoverflow.com/a/33024979
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def answers():
    global tagged_sentences_universal, test_data_universal, \
           train_data_universal, model, test_size, train_size, ttags, \
           correct, incorrect, accuracy, \
           good_tags, bad_tags, answer4b, answer5

    # Load the Brown corpus with the Universal tag set.
    tagged_sentences_universal = brown.tagged_sents(categories='news', tagset='universal')

    # Divide corpus into train and test data.
    test_size = 1000
    train_size = len(tagged_sentences_universal) - test_size # fixme

    test_data_universal = tagged_sentences_universal[:test_size] # fixme
    train_data_universal = tagged_sentences_universal[-train_size:] # fixme

    # Create instance of HMM class and initialise the training and test sets.
    model = HMM(train_data_universal, test_data_universal)

    # Train the HMM.
    model.train()

    # Inspect the model to see if emission_PD and transition_PD look plausible
    print('states: %s\n'%model.states)
    # Add other checks
    # check the conditions
    # print("The conditions of emission PD are: \n" + str(model.emission_PD.conditions()))
    # print("The conditions of transition_PD are: \n" + str(model.transition_PD.conditions()))
    # check with tiny examples
    # print('P(executive|NOUN) = ', model.emission_PD['NOUN'].prob('executive'))
    # print('P(term-end|NUM) =', model.emission_PD['NUM'].prob('term-end'))
    # print('P(NOUN|NOUN) =', model.transition_PD['NOUN'].prob('NOUN'))
    # print('P(ADJ|NOUN) =', model.transition_PD['NOUN'].prob('ADJ'))

    ######
    # Try the model, and test its accuracy [won't do anything useful
    #  until you've filled in the tag method
    ######
    s='the cat in the hat came back'.split()
    model.initialise(s[0])
    ttags = model.tag(s) # fixme
    print("Tag a trial sentence")
    print(list(zip(s,ttags)))

    # saving the incorrectly tagged test sents
    incorrectly_tagged_test_sents = []

    # check the model's accuracy (% correct) using the test set
    correct = 0
    incorrect = 0

    for sentence in test_data_universal:
        s = [word.lower() for (word, tag) in sentence]
        model.initialise(s[0])
        tags = model.tag(s)

        incorrect_sent = []
        is_incorrect = False
        for ((word,gold),tag) in zip(sentence,tags):
            if tag == gold:
                correct += 1
            else:
                incorrect += 1
                is_incorrect = True
            incorrect_sent.append((word, tag, gold))
        # Save the sentence if at least one tag is wrong
        if is_incorrect:
            incorrectly_tagged_test_sents.append(incorrect_sent)

    accuracy = float(correct) / (correct + incorrect) # fix me
    print('Tagging accuracy for test set of %s sentences: %.4f'%(test_size,accuracy))

    # Save first 10 incorrectly tagged test sents along with their correct version
    first_ten_incorrectly_tagged_sents = incorrectly_tagged_test_sents[:10]
    # Inspect these sentences and pick one
    chosen_sent = first_ten_incorrectly_tagged_sents[1]
    global tagged_sent, correct_sent
    tagged_sent = [(word, tag) for (word, tag, gold) in chosen_sent]
    correct_sent = [(word, gold) for (word, tag, gold) in chosen_sent]

    # Print answers for 4b and 5
    bad_tags, good_tags, answer4b = answer_question4b()
    print('\nAn incorrect tagged sequence is:')
    print(bad_tags)
    print('The correct tagging of this sentence would be:')
    print(good_tags)
    print('\nA possible reason why this error may have occurred is:')
    print(answer4b[:280])
    answer5=answer_question5()
    print('\nFor Q5:')
    print(answer5[:500])

if __name__ == '__main__':
    answers()
