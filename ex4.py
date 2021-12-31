import nltk
import numpy as np
from nltk.corpus import dependency_treebank

from collections import namedtuple
from networkx import DiGraph
from networkx.algorithms import maximum_spanning_arborescence
import random

nltk.download("dependency_treebank")

Arc = namedtuple('Arc', ['head', 'tail', 'weight'])

TAGS = ['LS', 'TO', 'VBN', "''", 'WP', 'UH', 'VBG', 'JJ', 'VBZ', '--', 'VBP', 'NN', 'DT', 'PRP', ':', 'WP$', 'NNPS',
        'PRP$', 'WDT', '(', ')', '.', ',', '``', '$', 'RB', 'RBR', 'RBS', 'VBD', 'IN', 'FW', 'RP', 'JJR', 'JJS', 'PDT',
        'MD', 'VB', 'WRB', 'NNP', 'EX', 'NNS', 'SYM', 'CC', 'CD', 'POS']

ITERATIONS = 2
LEARNING_RATE = 1


def load_data_and_preprocess():
    all_sents = dependency_treebank.parsed_sents()
    train_sents = all_sents[:int(len(all_sents) * 0.9)]
    test_sents = all_sents[int(len(all_sents) * 0.9):]
    return all_sents, train_sents, test_sents


def create_words_tags_index_dict():
    words = set(dependency_treebank.words())
    tags = TAGS
    index = 0
    words_dict = {}
    for word in words:
        words_dict[word] = index
        index += 1
    words_dict[None] = index
    index = 0
    tags_dict = {}
    for tag in tags:
        tags_dict[tag] = index
        index += 1
    tags_dict['TOP'] = index
    return words_dict, tags_dict


def feature_function(sent, words_dict, tags_dict):
    my_dict = {}
    for i in range(len(sent.nodes)):
        for j in range(len(sent.nodes)):
            word1 = sent.nodes[i]['word']
            word2 = sent.nodes[j]['word']
            tag1 = sent.nodes[i]['tag']
            tag2 = sent.nodes[j]['tag']
            row_word_index = (len(words_dict) * words_dict[word1]) + words_dict[word2]
            row_tags_index = (len(words_dict) ** 2 + 1) + (len(tags_dict) * tags_dict[tag1]) + tags_dict[tag2]
            my_dict[(i, j)] = (row_word_index, row_tags_index)
    return my_dict


def perceptron(sentences, words_dict, tags_dict):
    num_of_words = len(words_dict)
    num_of_tags = len(tags_dict)
    theta_k = np.zeros([num_of_words ** 2 + num_of_tags ** 2 + 2])
    theta = np.zeros([num_of_words ** 2 + num_of_tags ** 2 + 2])
    i = 0
    for r in range(ITERATIONS):
        for sent in sentences:
            feature_func = feature_function(sent, words_dict, tags_dict)
            max_tree_arcs = get_max_spanning_tree(sent, theta, num_of_words)
            predicted_vector = get_vector_from_arcs(max_tree_arcs, feature_func, num_of_words, num_of_tags)
            golden_vector = get_vector_from_gold_standard_tree(sent, feature_func, num_of_words, num_of_tags)
            theta += (LEARNING_RATE * (golden_vector - predicted_vector))
            theta_k += theta
            print(f'sentence number {i} of {len(sentences)} - {i / len(sentences)}%')
            i += 1
    return (1 / (ITERATIONS * (len(sentences) ** 2))) * theta_k


def predict(test_sentences, theta, num_of_words):
    for sent in test_sentences:
        get_max_spanning_tree(sent, theta, num_of_words)


def get_max_spanning_tree(sent, theta, num_of_words):
    """
    Wrapper for the networkX min_spanning_tree to follow the original API
    """

    arcs = []
    for i in range(len(sent.nodes)):
        for j in range(len(sent.nodes)):
            arcs.append(Arc(i, j, theta[i * num_of_words + j]))
    G = DiGraph()
    for arc in arcs:
        G.add_edge(arc.head, arc.tail, weight=arc.weight)
    ARB = maximum_spanning_arborescence(G)
    result = {}
    headtail2arc = {(a.head, a.tail): a for a in arcs}
    for edge in ARB.edges:
        tail = edge[1]
        result[tail] = headtail2arc[(edge[0], edge[1])]
    return result


def create_arc_list(sent, theta, num_of_words):
    arc_list = []
    for i in range(len(sent.nodes)):
        for j in range(len(sent.nodes)):
            arc_list.append(Arc(i, j, theta[i * num_of_words + j]))
    return arc_list


def get_vector_from_arcs(arcs, feature_function, num_of_words, num_of_tags):
    indices_pairs = []
    for arc in arcs:
        indices_pairs.extend(feature_function[(arcs[arc].head, arcs[arc].tail)])
    vector = np.zeros((num_of_words ** 2) + (num_of_tags ** 2) + 2)
    for i in indices_pairs:
        vector[i] = 1
    return vector


def get_vector_from_gold_standard_tree(sent, feature_function, num_of_words, num_of_tags):
    indices_pairs = []
    for node in sent.nodes:
        i = sent.nodes[node]['address']
        for j in sent.nodes[node]['deps']['']:
            indices_pairs.extend(feature_function[(i, j)])
    vector = np.zeros((num_of_words ** 2) + (num_of_tags ** 2) + 2)
    for i in indices_pairs:
        vector[i] = 1
    return vector


if __name__ == '__main__':
    all_sents, train_sents, test_sents = load_data_and_preprocess()
    random.shuffle(train_sents)
    words_dict, tags_dict = create_words_tags_index_dict()
    predicted_theta = perceptron(train_sents, words_dict, tags_dict)
    predictions = predict(test_sents, predicted_theta, len(words_dict))
