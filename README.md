# Topic Specificity

**Topic Specificity** is a measure for evaluating the concentration and distinctness of topics within a document corpus. It quantifies how well each topic stands out from the background noise by comparing document-level topic weights against a corpus-wide threshold. A higher specificity score indicates a topic that is more clearly defined and less diffuse across documents.

This package implements the specificity measure described in:

> **Yuan, M., Lin, P., Rashidi, L., & Zobel, J. (2023).** “Assessment of the Quality of Topic Models for Information Retrieval Applications.” *Proceedings of the 9th ACM SIGIR International Conference on the Theory of Information Retrieval (ICTIR ’23)*, 265–274. DOI: [10.1145/3578337.3605118](https://doi.org/10.1145/3578337.3605118)

---

## Installation

You can install this package directly using pip:

```bash
pip install topic-specificity
```

## Usage Examples

Below are some quick examples showing how to calculate topic specificity scores and Z-scores.

### 1. Calculate Specificity for All Topics

```python
from topic_specificity import calculate_specificity_for_all_topics
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary

# — assume you have preprocessed texts —
texts = [["human", "interface", "computer"], ["survey", "user", "computer", "system"]]
dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# Train a simple LDA model
lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=2, random_state=42)

# Compute specificity scores
spec_scores = calculate_specificity_for_all_topics(
    model=lda,
    corpus=corpus,
    mode='lda',
    threshold_mode='gmm',         # options: 'median', 'percentile', 'gmm'
    specificity_mode='sqrt'       # options: 'diff', 'sqrt'
)

print("Specificity scores:", spec_scores)
```

### 2. Calculate Z-Scores (Effect Sizes)

```python

from topic_specificity import calculate_Zi_scores

# Using the same LDA model & corpus as above:
z_scores = calculate_Zi_scores(
    model=lda,
    corpus=corpus,
    mode='lda'
)

print("Z-scores per topic:", z_scores)

```

### 3. LSA or HDP Models
The same functions work for LSA or HDP--just change the mode:
```python
from gensim.models.lsimodel import LsiModel
from gensim.models.hdpmodel import HdpModel

# Example for LSA:
lsa = LsiModel(corpus=corpus, id2word=dictionary, num_topics=2)
lsa_scores = calculate_specificity_for_all_topics(lsa, corpus, mode='lsa',
                                                  threshold_mode='median',
                                                  specificity_mode='diff')
print("LSA specificity:", lsa_scores)

# Example for HDP:
hdp = HdpModel(corpus=corpus, id2word=dictionary)
hdp_scores = calculate_specificity_for_all_topics(hdp, corpus, mode='hdp',
                                                  threshold_mode='percentile',
                                                  specificity_mode='sqrt')
print("HDP specificity:", hdp_scores)

```

## Citation

If you use **topic\_specificity** in your work, please cite:

```bibtex
@inproceedings{yuan2023assessment,
  author    = {Yuan, Meng and Lin, Pauline and Rashidi, Lida and Zobel, Justin},
  title     = {Assessment of the Quality of Topic Models for Information Retrieval Applications},
  booktitle = {Proceedings of the 9th ACM SIGIR International Conference on the Theory of Information Retrieval (ICTIR ’23)},
  year      = {2023},
  location  = {Taipei, Taiwan},
  pages     = {265--274},
  doi       = {10.1145/3578337.3605118},
}
```

DOI: [https://doi.org/10.1145/3578337.3605118](https://doi.org/10.1145/3578337.3605118)


