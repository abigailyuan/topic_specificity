"""
Topic Specificity Calculation Module.

Supports LDA, HDP, and LSA topic models with multiple thresholding methods.
"""

import os
import pickle
import numpy as np
from sklearn.mixture import GaussianMixture

def get_topic_weight_median(topic_weights):
    """Return the median of topic weights."""
    sorted_weights = np.sort(topic_weights)
    return sorted_weights[len(sorted_weights) // 2]


def get_96_percentile(topic_weights):
    """Return the 96th percentile (was previously labeled as 75_quartile)."""
    sorted_weights = np.sort(topic_weights)
    index = int(len(sorted_weights) * 0.96)
    return sorted_weights[index]


def get_threshold_from_gmm(topic_weights):
    """
    Fit a 2-component Gaussian Mixture Model to the topic weights and
    return the background threshold: mean + 2*std of the lower-mean component.
    """
    topic_weights = topic_weights.reshape(-1, 1)
    mixture = GaussianMixture(n_components=2, covariance_type="full").fit(topic_weights)
    means = mixture.means_.flatten()
    stds = np.sqrt(mixture.covariances_).flatten()

    # Background = component with the lower mean
    if means[0] <= means[1]:
        bi = means[0] + 2 * stds[0]
    else:
        bi = means[1] + 2 * stds[1]
    return bi


def filter_weights_above_threshold(threshold, topic_weights):
    """Return the list of weights above the threshold."""
    return [w for w in topic_weights if w > threshold]


def count_weights_above_threshold(threshold, topic_weights):
    """Return the count of weights above the threshold."""
    return np.sum(np.array(topic_weights) > threshold)


def calculate_specificity_score(bi, Vi, topic_weights):
    """
    Calculate the specificity score based on the formula:
    sqrt(sum((w - bi)^2)) / Vi scaled by (1 / (1 - bi)).
    """
    if Vi == 0:
        return 0  # Avoid division by zero

    squared_diffs = np.sum([(w - bi) ** 2 for w in topic_weights])
    specificity_score = np.sqrt(squared_diffs) / Vi
    scaled_score = specificity_score / (1 - bi)
    return scaled_score


def get_topic_distribution_lda(model, corpus):
    """Get topic distribution for each document using an LDA model."""
    num_topics = model.get_topics().shape[0]
    corpus_dist = np.zeros((len(corpus), num_topics))
    for i, doc in enumerate(corpus):
        dist = model.get_document_topics(doc, minimum_probability=0)
        for topic, prob in dist:
            corpus_dist[i][topic] = prob
    return corpus_dist


def get_topic_distribution_hdp(model, corpus, num_topics):
    """Get topic distribution for each document using an HDP model."""
    corpus_dist = np.zeros((len(corpus), num_topics))
    for i, doc in enumerate(corpus):
        dist = model[doc]
        for topic, prob in dist:
            if topic < num_topics:
                corpus_dist[i][topic] = prob
    return corpus_dist

def normalize_vector(vector):
    """Normalize a NumPy array so it sums to 1."""
    total = np.sum(vector)
    if total == 0:
        return vector
    return vector / total

def calculate_myui(weights, bi, Vi):
    """Mean of (weight - threshold)."""
    if Vi == 0:
        return 0
    return np.sum([(w - bi) for w in weights]) / Vi

def get_topic_distribution_lsa(model, corpus, num_topics):
    """Get normalized topic distribution for each document using an LSA model."""
    corpus_dist = np.zeros((len(corpus), num_topics))
    vectorized_corpus = model[corpus]

    for doc_id, doc in enumerate(vectorized_corpus):
        for topic, weight in doc:
            corpus_dist[doc_id][topic] = weight

    # Offset to make all weights non-negative
    global_min = corpus_dist.min()
    corpus_dist += -global_min

    # Normalize along topics for each document
    for row in range(len(corpus_dist)):
        corpus_dist[row] = normalize_vector(corpus_dist[row])

    return corpus_dist


def calculate_specificity_for_all_topics(model, corpus, mode, threshold_mode, specificity_mode, dist_override=None):
    """
    Calculate specificity scores for each topic in the corpus.

    - mode: 'lda', 'lsa', 'hdp'
    - threshold_mode: 'median', 'percentile', 'gmm'
    - specificity_mode: 'diff', 'sqrt'
    """
    scores = []
    dist_file = 'topic_distribution.pkl'
    save_dir = 'Results/wiki/3/'

    # Load or compute topic distributions
    if dist_override is not None:
        topic_distributions = dist_override
    elif dist_file in os.listdir(save_dir):
        topic_distributions = pickle.load(open(os.path.join(save_dir, dist_file), 'rb'))
    else:
        if mode == 'lsa':
            topic_distributions = get_topic_distribution_lsa(model, corpus)
        elif mode == 'lda':
            topic_distributions = get_topic_distribution_lda(model, corpus)
        elif mode == 'hdp':
            num_topics = model.get_topics().shape[0]
            topic_distributions = get_topic_distribution_hdp(model, corpus, num_topics)
        # Optionally save: pickle.dump(topic_distributions, open(os.path.join(save_dir, dist_file), 'wb'))

    num_topics = topic_distributions.shape[1]
    for topic_idx in range(num_topics):
        topic_weights = topic_distributions[:, topic_idx]

        # Select threshold method
        if threshold_mode == 'median':
            bi = get_topic_weight_median(topic_weights)
        elif threshold_mode == 'percentile':
            bi = get_96_percentile(topic_weights)
        elif threshold_mode == 'gmm':
            bi = get_threshold_from_gmm(topic_weights)
        else:
            raise ValueError(f"Unknown threshold_mode: {threshold_mode}")

        Vi = count_weights_above_threshold(bi, topic_weights)
        Di = filter_weights_above_threshold(bi, topic_weights)

        # Select specificity calculation method
        if specificity_mode == 'diff':
            myui = calculate_myui(Di, bi, Vi)
        elif specificity_mode == 'sqrt':
            myui = calculate_myui_sqrt(Di, bi, Vi)
        else:
            raise ValueError(f"Unknown specificity_mode: {specificity_mode}")

        # Final specificity score
        specificity_score = myui / (1 - bi) if (1 - bi) != 0 else 0
        scores.append(specificity_score)

    return scores


def calculate_myui_sqrt(weights, bi, Vi):
    """Square root of mean squared (weight - threshold)."""
    if Vi == 0:
        return 0
    return np.sqrt(np.sum([(w - bi) ** 2 for w in weights]) / Vi)

def calculate_variance(bi, myui, Vi, weights):
    """Calculate variance of (weight - bi - myui)."""
    if Vi <= 1:
        return 0
    return np.sum([(w - bi - myui) ** 2 for w in weights]) / (Vi - 1)


def calculate_Zi_scores(model, corpus, mode, dist_override=None):
    """
    Calculate Z-scores for each topic (effect size).
    """
    Z_scores = []
    dist_file = 'topic_distribution.pkl'
    save_dir = f'results/{mode.upper()}/'

    # Load or compute topic distributions
    if dist_override is not None:
        topic_distributions = dist_override
    elif dist_file in os.listdir(save_dir):
        topic_distributions = pickle.load(open(os.path.join(save_dir, dist_file), 'rb'))
    else:
        if mode == 'lsa':
            topic_distributions = get_topic_distribution_lsa(model, corpus)
        elif mode == 'lda':
            topic_distributions = get_topic_distribution_lda(model, corpus)
        elif mode == 'hdp':
            num_topics = model.get_topics().shape[0]
            topic_distributions = get_topic_distribution_hdp(model, corpus, num_topics)
        pickle.dump(topic_distributions, open(os.path.join(save_dir, dist_file), 'wb'))

    num_topics = topic_distributions.shape[1]
    for topic_idx in range(num_topics):
        topic_weights = topic_distributions[:, topic_idx]
        bi = get_threshold_from_gmm(topic_weights)
        Vi = count_weights_above_threshold(bi, topic_weights)

        if Vi < 2:
            Z_scores.append('N/A')
            continue

        Di = filter_weights_above_threshold(bi, topic_weights)
        myui = calculate_myui(Di, bi, Vi)
        var = calculate_variance(bi, myui, Vi, Di)

        if var == 0:
            Z_scores.append('N/A')
        else:
            Zi = myui / np.sqrt(var)
            Z_scores.append(Zi)

    return Z_scores
