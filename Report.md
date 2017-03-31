# Machine Learning Nanodegree
## Capstone Report
Roman Levitas
March 2017

## Transferred Learning: Is Physics the Common Thread?

### Definition

# Overview

The aim of this investigation will be to build a text classifier. The twist is that the classifier will be trained on topics independent of the testing set in theme, making this a classic Natural Language Processing (NLP) problem with a transferred learning caveat.

The input data to be analyzed is a subset of the Stack Exchange Data Dump [1] published on December 15, 2016 focusing on 6 topics in particular: biology, cooking, cryptography, diy, robotics, and travel.
The testing set is a 7th topic--physics.
In effect, the classifier to be designed will have knowledge of 6 seemingly independent topics through which we can investigate the idea of there being a common thread or unifying theme, which is physics.

The history of Stack Exchange's question-and-answer format dates back to as recent as 2008 when StackOverflow was created, the originating leg of the Stack Exchange network, which allows users to crowdsource knowledge on the topic of computer science/software engineering.

The data dump contains the titles, text and tags of Stack Exchange questions in the form of a comma separated value (CSV) document where each row is a separate question. The Titles and text of the question will be considered as the input set and the tags, or classification, is what will be predicted via the classifier.

[1] https://archive.org/details/stackexchange

### Problem Statement
The problem at hand is defined by the Kaggle team's competition title: 'Transfer Learning on Stack Exchange Tags' [3] which aims to "Predict tags from models trained on unrelated topics". Specifically, predicting tags for physics questions after training the text classifier on questions provided in the 6 different fields mentioned above.

Underlying in this approach is the assumption that there is a unifying thread, so the investigation is itself an exercise into a problem without a definite answer. However, if there is a correlation that can be found, it can shed light onto the longstanding question: Is Physics at the heart of everything?

Predictions can be compared against the physics questions' actual, assigned tags in the test set and thus the classification model can be ranked on the correctness of its categorization using metrics discussed further.

[3] https://www.kaggle.com/c/transfer-learning-on-stack-exchange-tags

### Evaluation Metrics
The evaluation metric for this competition is Mean F1-Score [5]. The F1 score measures accuracy using the statistical notions of precision (p) and recall (r). Precision is the ratio of true positives (tp) to all predicted positives (tp + fp). Recall is the ratio of true positives to all actual positives (tp + fn) where positives are correctly classified tags.

The F1 score is given by:
  F1 = 2pr/p+r
where
  p=tp/tp+fp  and  r=tp/tp+fn


In the multi-class and multi-label case, Mean F1-Score is the weighted average of the F1 score of each class. [6]

Example:
[Quantum-Mechanics, Electron, Current] => [ Electron, Ampere, Quantum-Mechanics, Voltage]


[5] https://www.kaggle.com/c/transfer-learning-on-stack-exchange-tags#evaluation
[6] http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html

### Analysis

# Data Exploration
The input dataset is 6 separate CSV files of questions and their tags as strings in these fields: biology, cooking, cryptography, diy, robotics, and travel [4]. Each row of the CSV contains the title, text, and associated tags of a question. The export of the data from StackExchange supported html markdown for the text column so there will be a data cleaning step to take into account markdown formatting and html tags/element removal.

The columns in the csv are defined as follows:
`'id', 'title', 'content', 'tags'`

`id` field will be ignored as the python library used for importing the files will take care of indexing itself.


3 rows from the robotics data set are printed below in their raw form:

id                                                         1
title      What is the right approach to write the spin c...
content    <p>Imagine programming a 3 wheel soccer robot....
tags                                          soccer control

id                                                         2
title      How can I modify a low cost hobby servo to run...
content    <p>I've got some hobby servos (<a href="http:/...
tags                                         control rcservo

id                                                         3
title      What useful gaits exist for a six legged robot...
content    <p><a href="http://www.oricomtech.com/projects...
tags                                               gait walk

The number of questions/rows for each of the 6 sets varies as so:

biology: 13,196
cooking: 15,404
travel: 19,279
robotics: 2,771
crypto: 10,432
diy: 25,918

The testing data in the 7th set has 81,926 entries on physics.

[4] https://www.kaggle.com/c/transfer-learning-on-stack-exchange-tags/data

# Exploratory Visualization
One measure of how important a word may be in a text collection is its term frequency (tf), how frequently a word occurs in a document. There are words in a document, however, that occur many times but may not be significant; in English, these are words like “the”, “is”, “of”, and so forth

Another approach is to look at a term’s inverse document frequency (idf), which decreases the weight for commonly used words and increases the weight for words that are not used very much in a collection of documents. This can be combined with term frequency to calculate a term’s tf-idf,
which is a way to score the importance of words in a document based on how frequently they appear across multiple documents, or in this case, questions.

`Histograms for tf-idf`
`wordcloud`

# Algorithms and Techniques
To solve the problem posed by the Kaggle competition a Decision Tree classifier will be used to train it on the dataset of 6 fields and see if it can give reasonable assignment of tags on the testing set which is in a different, 7th, field--physics.

First, the data set is preprocessed by reducing the content text of it's markdown properties.
Furthermore, by utilizing classic NLP approaches such as removing stopwords, converting case to a convention (lower), and removing punctuation, the data is transformed to allow for more meaningful results.

As tfidf was used in the previous section to extract the dominating and significant features of our input data and visualize it via histograms, it will be utilized by the decision tree classifier as well. This is done with
`tfidf vectorizer`

In order to gauge the general metrics of the classifier, the training sets are individually cross validated using train/test splits.
Cross validation grid search also allows a wide range of tunable parameters to be optimized by trying different combinations of values.

### Benchmark Model
The benchmark model for the classifier will be running the model on unseen physics questions and analyzing their text and title in order to predict tags (actual tags are provided for testing). The results can be evaluated as true positives (tp) and false positives (fp)--together all predicted positives, and true positives with false negatives (fn)--all actual positives.
These categorizations allow for empirical analysis of our model and can be compared across implementations of classification algorithms.

The top ten F1 scores of the competition leaderboard range between 0.2 - 0.29 (excluding an outlier in first place with a score of .67).
To be able to predict tags with an F1 score above 0.2 would be satisfactory, above 0.25 would be great, and above 0.3 would be outstanding.





### Project Design
The proposed project will be to investigate the potential correlation of physics to 6 other areas and see if the knowledge can be generalized. The correlation may be low due to a lack of cross over in the fields or due to a weakness in the algorithm finding the correlation. The latter can be optimized to make sure that it is not the failing link.

Before getting started, it would be wise to examine the provided data and possibly strip it of its HTML formatting so as to prevent analyzing <p></p> tags as something meaningful and apply NLP techniques mentioned in the Solution Statement.

Further, the 3 various classification algorithms to be explored (kNN, SVMs, Decision Trees) provide variety in how they utilize the data--some more prone to over or under fitting or have parameters that can be fine tuned.
For example, for decision trees, pruning can be used to avoid overfitting. For kNN, the value of k should be reasonable for the provided data. GridSearchCV will be utilized in addressing this.

Each of the trained models will be run against a test CSV dataset which has physics questions and their corresponding tags in order to calculate the F1 score in accordance to evaluation metrics and the model's tag predictions. In the case that correlation is found to be extremely weak in testing, optimized tagging classifiers will still have been developed that can run independently in their own field of knowledge.

A theory exists stating that any 2 people can be connected through 6 traversals on a graph and is known as 6 degrees of Separation [7].
Can the same be applied to 6 random topics and have physics be the common connection between them all?
This investigation sets to find out.

[7] https://en.wikipedia.org/wiki/Six_degrees_of_separation
