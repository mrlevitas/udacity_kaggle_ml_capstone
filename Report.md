# Machine Learning Nanodegree
## Capstone Report
Roman Levitas
March 2017

## Transferred Learning Natural Language Processing Report

### Definition

# Overview

The aim of this investigation will be to build a text classifier, but the twist is that the classifier will be trained on topics independent of the testing set, making this a classic Natural Language Processing (NLP) problem with a transferred learning approach.

The data to be analyzed is a subset of the Stack Exchange Data Dump [1] published on December 15, 2016 focusing on 6 topics in particular: biology, cooking, cryptography, diy, robotics, and travel.
The testing set is a 7th topic--physics.
In effect, the classifier to be designed will have knowledge of 6 seemingly independent topics through which we can investigate the idea of there being a common thread or unifying theme if looked at from the perspective of physics.

The history of Stack Exchange's question-and-answer format dates back to as recent as 2008 when StackOverflow was created, the originating leg of the Stack Exchange network, which allows users to crowdsource knowledge on the topic of computer science/software engineering.

The data dump contains the titles, text, (the input set) and tags (which is what we will be predicting via the classifier) of Stack Exchange questions in the form of a comma separated value (CSV) document.

[1] https://archive.org/details/stackexchange

### Problem Statement
The problem at hand is defined by the Kaggle team's competition title: 'Transfer Learning on Stack Exchange Tags' [3] which aims to "Predict tags from models trained on unrelated topics". Specifically, predicting tags for physics questions after training the classifier on questions provided in the 6 different fields mentioned above.

Underlying in this approach is the presumption that physics is the unifying thread, so the investigation is itself an exercise into a problem without a definite answer but if there is a correlation that can be found, it can certainly shed light and illuminate the grey area in question: Is Physics at the heart of eveything?

Predictions can be compared against the physics questions' actual tags and thus the model can be ranked on the correctness of its categorization using metrics discussed further.

[3] https://www.kaggle.com/c/transfer-learning-on-stack-exchange-tags

### Evaluation Metrics
The evaluation metric for this competition is Mean F1-Score [5]. The F1 score measures accuracy using the statistical notions of precision (p) and recall (r). Precision is the ratio of true positives (tp) to all predicted positives (tp + fp). Recall is the ratio of true positives to all actual positives (tp + fn).

The F1 score is given by:
  F1 = 2pr/p+r
where
  p=tp/tp+fp  and  r=tp/tp+fn


In the multi-class and multi-label case, Mean F1-Score is the weighted average of the F1 score of each class. [6]

Example:
Prediction => Actual
[Quantum-Mechanics, Electron, Current] => [ Electron, Ampere, Quantum-Mechanics, Voltage]
Quantum-Mechanics: tp
Electron: tp
Current: fp
!Ampere: fn
!voltage: fn

2 tp, 1fp, 2 fn

p = 2/3 , r = 1/2

F1= 4/7



[5] https://www.kaggle.com/c/transfer-learning-on-stack-exchange-tags#evaluation
[6] http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html

### Analysis

# Data Exploration
The input dataset is 6 separate CSV files for biology, cooking, cryptography, diy, robotics, and travel [4]. Each row of the data contains the title, text, and associated tags of a question. The export of the data from StackExchange supported html markdown for the text column so there will be a prerequisite data cleansing step to take into account markdown formatting and html tags/elements.

`biology.head(5)`

The training set contains ~25,000 entries for diy, ~8,600 for biology, ~10,400 for cryptography, ~2,700 for robotics, ~12,000 for travel, ~15,000 for cooking.

The testing data has ~82,000 entries on physics.

[4] https://www.kaggle.com/c/transfer-learning-on-stack-exchange-tags/data

# Exploratory Visualization
One measure of how important a word may be in a text collection is its term frequency (tf), how frequently a word occurs in a document. There are, however, words in a document that occur many times but may not be significant; in English, these are words like “the”, “is”, “of”, etc.

Another approach is to look at a term’s inverse document frequency (idf), which decreases the weight for commonly used words and increases the weight for words that are not used very much in a collection of documents. This can be combined with term frequency to calculate a term’s tf-idf,
which is a way to score the importance of words in a document based on how frequently it appears across multiple documents, or in our case, rows.

To get a sense of the questions, a histogram of the top 20 terms according to the tf-idf scoring of the titles from the biology CSV provides insight into their content. This is to get some intuition of what our classifier will be working with as input values, after a data cleaning step.
`Histogram for biology tf-idf`

Furthermore, the output values, or tags, are visualized below in a word-cloud which ranks the tags to be larger if it is more popular, or frequent.
`wordcloud for biology tf`


# Algorithms and Techniques
To solve the problem a Decision Tree classifier will be trained and used to see if it can give reasonable assignment of tags on the testing set.

As tfidf was used in the previous section to extract the dominating and significant features of our input data and visualize it via histograms, it will also be utilized by the decision tree classifier as well. This is done with Sci-kit learn python library by transforming the input data into useful numerical values that are easier for a classifier to consume.   

Cross validation grid search (`GridSearchCV()`) allows a wide range of tunable parameters to be optimized by trying different combinations of values.
For the TfidfVectorizer, the minimum & maximum data frequency (df) parameters can be used to create a range of acceptable term frequencies and cut off trivial terms (those that appear too infrequently and cannot be used to generalize patterns or vice versa--terms that appear too frequently, across all rows, and thus add no value).

The following values were used when experimenting and tuning the vectorizer:

tfidf__min_df : [ 0.001, 0.005, 0.01 , 0.05],
tfidf__max_df : [0.9, 0.95 , 0.975, 0.99]

where values lie between 0 and 1.0

The resultant term frequencies are analyzed by a `DecisionTreeClassifier()`

A decision tree is rooted in making choices that maximize the likelyhood of predicting if a term frequency is correlated to a tag.
The following parameters are iterated over for the classifier using GridSearchCV():

DT__criterion : ["gini", "entropy"]
DT__max_depth : [10, 20, 50, 100, 200, 500]

The criterion according to which the quality of a split is calculated when subsetting nodes of the tree into its children nodes varies between either "gini" impurity or the traditional "entropy" definition of information gain.

The concept of decision trees similar to the game of 20 questions, where each question tries to gain more information about the target answer. In this case, "good" questions can be defined mathematically through entropy to maximize the information gain at each question or node in the tree. To solidify what qualifies as "good" questions can be understood practically, without entropy: if playing the 20 quesitons game, one can start by asking a broad question to find out general classification (i.e. is it an object or living?) and narrowing in more as you get closer to the answer (if the answer to the previous question is living, we can ask--is it a human or other?).
Once these questions have been asked and answers associated, the tree is built and one must simply follow it with a set of testing text (answering each question along the path as you go) to see if a tag is pertinent.

DecisionTrees are vulnerable to overfitting and thus the max_depth parameter is used to set the maximum depth of the classifier before it is pruned. If no parameter is given, the tree will continue to grow until all the leaf nodes are pure.


In order to gauge the general metrics of the classifier, the training sets are individually cross validated using train/test splits.



### Benchmark
The benchmark model for the classifier will be running the model on unseen physics questions and analyzing their text and title in order to predict tags (actual tags are provided for testing). The results can be evaluated as true positives (tp) and false positives (fp)--together all predicted positives, and true positives with false negatives (fn)--all actual positives.
These categorizations allow for empirical analysis of our model and can be compared across implementations of classification algorithms.

The top ten F1 scores of the competition leaderboard range between 0.2 - 0.29 (excluding an outlier in first place with a score of .67).
To be able to predict tags with an F1 score above 0.2 would be satisfactory, above 0.25 would be great, and above 0.3 would be outstanding.

## Methodology

# Data Preprocessing

First, the data set is preprocessed by stripping the question text of it's markdown properties and removal of wrapping HTML tags and code snippets.
Then, by utilizing classic NLP approaches such as removing stopwords, converting case to a convention (lower), removing punctuation/non-alphabet characters, and reducing permutations of the same word by using its linguistic stem, the data is transformed to allow for more meaningful results. For implementation, see `clean_data( raw_data )` function.

Tags are analyzed as strings, separated by spaces (preserving hyphenated words), and are set to an array of strings. They are further encoded into bit-arrays where the length of the array is the number of unique combination of tags, and a 1 in a specified place represents its presence in a question.


# Implementation
To get set up, two Python libraries are used predominantly throughout the investigation: `numpy` for its linear algebra functionality and `pandas` for CSV file I/O, in-memory DataFrame matrix representation, and data processing.

CSV's are imported into pandas DataFrames:
  `biology = pd.read_csv("data/biology.csv")
   travel = ...`
and are combined and stored in a dictionary:

`  df_hash = {
      "biology": biology,
      "travel": travel,
      "diy": diy,
      "cooking": cooking,
      "crypto": crypto,
      "robotics": robotics
  }`

The implementation of the data preprocessing is aggregated into the `clean_data()` method which takes raw_data from the Dataframes as a parameter.
The method relies on two more linguistic Python libraries used commonly to accomplish the goals laid out in ##Data Preprocessing section: `bs4` (BeautifulSoup4) used for parsing out code snippets and html elements and `nltk` (Natural Language Toolkit) for lemmatizing/stemming words and only retaining words with meaning.

The remainder of investigation focuses on machine learning steps and utilizes the scikit-learn (aka SKlearn) Python library extensively.
For ease of use and chaining data transformations, a pipeline is created to funnel preprocessed data through a `TfidfVectorizer()` for both the questions' titles and text (using the `Selector` class as a custom transformer), and their resultant values combined using a `FeatureUnion`.

These Tfidf values are used to construct decision trees using the `DecisionTreeClassifier` wrapped in a `OneVsRestClassifier` for each tag.

This constructed pipeline is then fit through `GridSearchCV` (cross validation) where a variety of parameters can be applied for each transform and the best combination is reported.

A `MultiLabelBinarizer` is used to accomplish encoding of tags


### Project Design
For decision trees, pruning can be used to avoid overfitting. GridSearchCV will be utilized in addressing this.
