# IMDb Sentiment Analysis: Lexicon-Based vs. Machine Learning

Developed a natural language processing (NLP) pipeline to analyze sentiment in IMDb movie reviews using **four sentiment lexicons and a supervised machine learning classifier**.

The project compares AFINN, Bing, NRC, and SentiWordNet for lexicon-based sentiment scoring, then trains a TF-IDF-based Logistic Regression model using automatically generated sentiment labels.

## Key Features

- Processed a balanced dataset of 2,000 IMDb movie reviews sampled from the original 50,000-review dataset.
- Implemented text preprocessing using R and Python, including HTML removal, tokenization, text normalization, and stop-word filtering.
- Compared four sentiment lexicons: AFINN, Bing, NRC, and SentiWordNet.
- Analyzed sentiment polarity and eight emotion categories using NRC.
- Applied NLTK part-of-speech tagging and lemmatization for SentiWordNet scoring.
- Generated automatic sentiment labels using the Bing lexicon.
- Extracted up to 5,000 TF-IDF features using unigrams and bigrams.
- Trained a Logistic Regression classifier with an 80/20 stratified train-test split.
- Evaluated predictions using accuracy, precision, recall, F1 score, and confusion matrices.
- Investigated class imbalance, lexicon bias, and limitations of automatically generated labels.

## Model Performance

**Lexicon Agreement with Original IMDb Labels**

| Lexicon | Agreement |
|---|---|
| AFINN | 74.4% |
| Bing | 73.8% |
| NRC | 65.6% |
| SentiWordNet | 67.6% |

**TF-IDF + Logistic Regression**

| Metric | Result |
|---|---|
| Accuracy vs. Bing labels | 74.7% |
| Weighted F1 Score | 0.706 |
| Accuracy vs. original IMDb labels | 61.8% |
| Training Samples | 1,486 |
| Testing Samples | 372 |

The classifier achieved **74.7% accuracy against automatically generated Bing labels**. However, accuracy against the original IMDb sentiment labels was 61.8%, highlighting the limitations of weak supervision and lexicon-based labeling.

## Technologies

**Languages:** Python, R

**NLP:** NLTK, SentiWordNet, AFINN, Bing, NRC, TF-IDF

**Machine Learning:** Scikit-learn, Logistic Regression

**Data Analysis:** Pandas, Tidyverse, Tidytext, Matplotlib, ggplot2

**Environment:** Jupyter Notebook, Google Colab, RStudio

## Key Takeaways

This project demonstrates how lexicon-based sentiment analysis can be combined with supervised machine learning to build an NLP classification pipeline.

The comparative evaluation also highlights challenges involving class imbalance, sentiment ambiguity, contextual interpretation, and differences between automatically generated labels and human annotations.
