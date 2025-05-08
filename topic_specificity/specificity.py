"""
Topic Specificity Calculation Module.

Based on:
  Yuan, M., Lin, P., Rashidi, L., & Zobel, J. (2023).
  Assessment of the Quality of Topic Models for Information Retrieval Applications.
  In ICTIR ’23 (pp. 265–274). ACM.
  https://doi.org/10.1145/3578337.3605118

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
    """Return the 96th percentile of topic weights."""
    sorted_weights = np.sort(topic_weights)
    index = int(len(sorted_weights) * 0.96)
    return sorted_weights[index]


def get_threshold_from_gmm(topic_weights):
    """
    Fit a 2-component Gaussian Mixture Model to the topic weights and
    return the background threshold: mean + 2*std of the lower-mean component.
    """
    reshaped = topic_weights.reshape(-1, 1)
    mixture = GaussianMixture(n_components=2, covariance_type='full').fit(reshaped)
    means = mixture.means_.flatten()
    stds = np.sqrt(mixture.covariances_).flatten()
    # background is the component with lower mean
    if means[0] <= means[1]:
        return means[0] + 2 * stds[0]
    else:
        return means[1] + 2 * stds[1]


def filter_weights_above_threshold(threshold, topic_weights):
    """Return weights above the threshold."""
    return [w for w in topic_weights if w > threshold]


def count_weights_above_threshold(threshold, topic_weights):
    """Return count of weights above the threshold."""
    return int(np.sum(np.array(topic_weights) > threshold))


def normalize_vector(vector):
    """Normalize a NumPy vector to sum to 1."""
    total = np.sum(vector)
    return vector / total if total != 0 else vector


def calculate_myui(weights, bi, Vi):
    """Mean of (weight - threshold)."""
    return np.sum([(w - bi) for w in weights]) / Vi if Vi else 0


def calculate_myui_sqrt(weights, bi, Vi):
    """Square root of mean squared (weight - threshold)."""
    return np.sqrt(np.sum([(w - bi) ** 2 for w in weights]) / Vi) if Vi else 0


def calculate_specificity_for_all_topics(
    model,
    corpus,
    mode,
    threshold_mode,
    specificity_mode,
    dist_override=None,
    dist_file='topic_distribution.pkl',
    save_dir=None
):
    """
    Calculate specificity scores for each topic in the corpus.
    mode: 'lda', 'lsa', 'hdp'
    threshold_mode: 'median', 'percentile', 'gmm'
    specificity_mode: 'diff', 'sqrt'
    """
    save_dir = save_dir or os.getcwd()
    # load or compute distributions
    if dist_override is not None:
        distributions = dist_override
    elif dist_file in os.listdir(save_dir):
        distributions = pickle.load(open(os.path.join(save_dir, dist_file), 'rb'))
    else:
        if mode == 'lsa':
            distributions = get_topic_distribution_lsa(model, corpus)
        elif mode == 'lda':
            distributions = get_topic_distribution_lda(model, corpus)
        elif mode == 'hdp':
            num_topics = model.get_topics().shape[0]
            distributions = get_topic_distribution_hdp(model, corpus, num_topics)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    scores = []
    num_topics = distributions.shape[1]
    for t in range(num_topics):
        weights = distributions[:, t]
        # threshold
        if threshold_mode == 'median':
            bi = get_topic_weight_median(weights)
        elif threshold_mode == 'percentile':
            bi = get_96_percentile(weights)
        elif threshold_mode == 'gmm':
            bi = get_threshold_from_gmm(weights)
        else:
            raise ValueError(f"Unknown threshold_mode: {threshold_mode}")
        Vi = count_weights_above_threshold(bi, weights)
        Di = filter_weights_above_threshold(bi, weights)
        # specificity
        if specificity_mode == 'diff':
            myui = calculate_myui(Di, bi, Vi)
        elif specificity_mode == 'sqrt':
            myui = calculate_myui_sqrt(Di, bi, Vi)
        else:
            raise ValueError(f"Unknown specificity_mode: {specificity_mode}")
        M_i = myui / (1 - bi) if (1 - bi) else 0
        scores.append(M_i)
    return scores


def get_topic_distribution_lda(model, corpus):
    """Get topic distribution per document using LDA model."""
    num_topics = model.get_topics().shape[0]
    dist = np.zeros((len(corpus), num_topics))
    for i, doc in enumerate(corpus):
        for topic, prob in model.get_document_topics(doc, minimum_probability=0):
            dist[i][topic] = prob
    return dist


def get_topic_distribution_hdp(model, corpus, num_topics):
    """Get topic distribution per document using HDP model."""
    dist = np.zeros((len(corpus), num_topics))
    for i, doc in enumerate(corpus):
        for topic, prob in model[doc]:
            if topic < num_topics:
                dist[i][topic] = prob
    return dist


def get_topic_distribution_lsa(model, corpus, num_topics):
    """Get normalized distribution per document using LSA model."""
    dist = np.zeros((len(corpus), num_topics))
    vecs = model[corpus]
    for i, doc in enumerate(vecs):
        for topic, weight in doc:
            dist[i][topic] = weight
    # offset and normalize
    dist += -dist.min()
    for i in range(len(dist)):
        dist[i] = normalize_vector(dist[i])
    return dist


def calculate_Zi_scores(model, corpus, mode, dist_override=None, dist_file='topic_distribution.pkl', save_dir=None):
    """
    Calculate Z-scores (effect size) for each topic.
    """
    save_dir = save_dir or os.getcwd()
    if dist_override is not None:
        distributions = dist_override
    elif dist_file in os.listdir(save_dir):
        distributions = pickle.load(open(os.path.join(save_dir, dist_file), 'rb'))
    else:
        if mode == 'lsa':
            distributions = get_topic_distribution_lsa(model, corpus)
        elif mode == 'lda':
            distributions = get_topic_distribution_lda(model, corpus)
        elif mode == 'hdp':
            num_topics = model.get_topics().shape[0]
            distributions = get_topic_distribution_hdp(model, corpus, num_topics)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        pickle.dump(distributions, open(os.path.join(save_dir, dist_file), 'wb'))
    Zs = []
    num_topics = distributions.shape[1]
    for t in range(num_topics):
        weights = distributions[:, t]
        bi = get_threshold_from_gmm(weights)
        Vi = count_weights_above_threshold(bi, weights)
        if Vi < 2:
            Zs.append('N/A')
            continue
        Di = filter_weights_above_threshold(bi, weights)
        myui = calculate_myui(Di, bi, Vi)
        var = calculate_variance(bi, myui, Vi, Di)
        Zs.append(myui / np.sqrt(var) if var else 'N/A')
    return Zs


def calculate_variance(bi, myui, Vi, weights):
    """Calculate variance of (weight - bi - myui)."""
    return np.sum([(w - bi - myui) ** 2 for w in weights]) / (Vi - 1) if Vi > 1 else 0
