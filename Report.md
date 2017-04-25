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
In effect, the classifier to be designed will have knowledge of 6 seemingly independent topics. The idea of there being a common thread or unifying theme if looked at from the perspective of physics among these topics is to be investigated.

The history of Stack Exchange's question-and-answer format dates back to as recent as 2008 when StackOverflow was created, the originating leg of the Stack Exchange network, which allows users to crowdsource knowledge on the topic of computer science/software engineering.

The data dump contains the titles, text, (the input set) and tags (which is what we will be predicting via the classifier) of Stack Exchange questions in the form of a comma separated value (CSV) document.

[1] https://archive.org/details/stackexchange

### Problem Statement
The problem at hand is defined by the Kaggle team's competition title: 'Transfer Learning on Stack Exchange Tags' [3] which aims to "Predict tags from models trained on unrelated topics". Specifically, predicting tags for physics questions after training the classifier on questions provided in the 6 different fields mentioned above.

Underlying in this approach is the presumption that physics is the unifying concept, so the investigation is itself an exercise into a problem without a definite answer but if there is a correlation that can be found, it can certainly shed light and illuminate the grey area in question: Is Physics at the heart of eveything?

Predictions can be compared against the physics questions' actual tags and thus the model can be ranked on the correctness of its categorization using metrics discussed below.

[3] https://www.kaggle.com/c/transfer-learning-on-stack-exchange-tags

###  Metrics
The evaluation metric for this competition is Mean F1-Score [5]. The F1 score measures accuracy using the statistical notions of precision (p) and recall (r). Precision is the ratio of true positives (tp) to all predicted positives (tp + fp). Recall is the ratio of true positives to all actual positives (tp + fn).

The F1 score is given by:
  F1 = 2pr/p+r
where
  p=tp/tp+fp  and  r=tp/tp+fn

In the multi-class and multi-label case, Mean F1-Score is the weighted average of the F1 score of each class. [6]

To better understand this, an example is given below:

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
The input dataset is 6 separate CSV files for biology, cooking, cryptography, diy, robotics, and travel [4]. Each row of the data contains the title, text, and associated tags of a question. The export of the data from StackExchange supported html markdown for the text column so there will be a prerequisite data cleaning step to take into account markdown formatting and html tags/elements.

The first five rows of the biology CSV are shown:
`biology.head(5)`

The training set contains ~25,000 entries for diy, ~8,600 for biology, ~10,400 for cryptography, ~2,700 for robotics, ~12,000 for travel, ~15,000 for cooking.

The testing data has ~82,000 entries on physics but lacks the tags, which are to be predicted.

[4] https://www.kaggle.com/c/transfer-learning-on-stack-exchange-tags/data

# Exploratory Visualization
One measure of how important a word in a text collection is its term frequency (tf), how frequently a word occurs in a document. There are, however, words in a document that occur many times but may not be significant; in English, these are words like “the”, “is”, “of”, etc.

Another approach is to look at a term’s inverse document frequency (idf), which decreases the weight for commonly used words and increases the weight for words that are not used very much in a collection of documents. This can be combined with term frequency to calculate a term’s tf-idf,
which is a way to score the importance of words in a document based on how frequently it appears across multiple documents, or in this case, rows/questions.

To get a sense of the questions, a histogram of the top 20 terms according to the tf-idf scoring of the titles from the biology CSV provides insight into their content. This provides some intuition as to what the classifier will be working with as input values, after a data cleaning step.
`Histogram for biology tf-idf`

Furthermore, the output values, or tags, are visualized below in a word-cloud which ranks the tags to be larger if it is more popular, or frequent.
`wordcloud for biology tf`


# Algorithms and Techniques
To solve the problem a Decision Tree classifier will be trained and used to see if it can give reasonable assignment of tags on the testing set.

As tfidf was used in the previous section to extract the dominating and significant features of our input data and visualize it via histograms, it will be utilized by the decision tree classifier as well. This is done with a python library by transforming the input data into useful numerical values that are easier for a classifier to consume.   

Furtermore, Cross validation grid search (`GridSearchCV()`) is used to wrap the classifier and allows a wide range of tunable parameters to be optimized by trying different combinations of values.
For the TfidfVectorizer, the minimum & maximum data frequency (df) parameters can be used to create a range of acceptable term frequencies and cut off trivial terms (those that appear too infrequently and cannot be used to generalize patterns or vice versa--terms that appear too frequently, across all rows, and thus add no value).

The resultant term frequencies are analyzed by a `DecisionTreeClassifier()` which is rooted in making choices that maximize the likelyhood of predicting if a term frequency is correlated to a tag.

The criterion according to which the quality of a split is calculated when subsetting nodes of the tree into its children nodes varies between either "gini" impurity or the traditional "entropy" definition of information gain.

The concept of decision trees similar to the game of '20 Questions', where each question tries to gain more information about the target answer. In this case, "good" questions can be defined mathematically through entropy to maximize the information gain at each question or node in the tree. To solidify what qualifies as "good" questions can be understood practically, without entropy: if playing the 20 quesitons game, one can start by asking a broad question to find out general classification (i.e. is it an object or living?) and narrowing in more as you get closer to the answer (if the answer to the previous question is living, we can ask--is it a human or other?).
Once these questions have been asked and answers associated, the tree is built and one must simply follow it with a set of testing text (answering each question along the path as you go) to see if a tag is pertinent.

DecisionTrees are vulnerable to overfitting and thus the max_depth parameter is used to set the maximum depth of the classifier before it is pruned. If no parameter is given, the tree will continue to grow until all the leaf nodes are pure.

In order to gauge the general metrics of the classifier, the training sets are individually cross validated using train/test splits with the CV parameter.


### Benchmark
The benchmark model for the classifier will be running the model on unseen physics questions and analyzing their text and title in order to predict tags (actual tags are provided for testing). The results can be evaluated as true positives (tp) and false positives (fp)--together all predicted positives, and true positives with false negatives (fn)--all actual positives.
These categorizations allow for empirical analysis of the model and can be compared across implementations of classification algorithms.

The top ten F1 scores of the competition leaderboard range between 0.2 - 0.29 (excluding an outlier in first place with a score of .67).
To be able to predict tags with an F1 score above 0.2 would be satisfactory, above 0.25 would be great, and above 0.3 would be outstanding.

## Methodology

# Data Preprocessing

First, the data set is preprocessed by stripping the question text of it's markdown properties and removal of wrapping HTML tags and code snippets.
Then, by utilizing classic NLP approaches such as removing stopwords, converting case to a convention (lower), removing punctuation/non-alphabet characters, and reducing permutations of the same word by using its linguistic stem, the data is transformed to allow for more meaningful results.

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

The implementation of data preprocessing is aggregated into the `clean_data()` method which takes raw_data from the Dataframes as a parameter.
The method relies on two more linguistic Python libraries used commonly to accomplish the goals laid out in ##Data Preprocessing section: `bs4` (BeautifulSoup4) used for parsing out code snippets and html elements and `nltk` (Natural Language Toolkit) for lemmatizing/stemming words and only retaining words with meaning.

The remainder of investigation focuses on machine learning steps and utilizes the scikit-learn (aka SKlearn) Python library extensively.
For ease of use and chaining data transformations, a pipeline is created to funnel preprocessed data through a `TfidfVectorizer()` for both the questions' titles and text (using the `Selector` class as a custom transformer to subset the DataFrames), and their resultant Tfidf values combined using a `FeatureUnion`.

These Tfidf values are used to construct decision trees using the `DecisionTreeClassifier` wrapped in a `OneVsRestClassifier` for each tag.

This constructed pipeline is then fit through `GridSearchCV` (cross validation) where a variety of parameters can be applied for each transform and the best combination is reported.

The following values were tried when experimenting and tuning the TfidfVectorizer:

  tfidf__min_df : [ 0.001, 0.005, 0.01 , 0.05],
  tfidf__max_df : [0.9, 0.95 , 0.975, 0.99]


while for the DecisionTreeClassifier parameters varied between:

  DT__criterion : ["gini", "entropy"]
  DT__max_depth : [10, 25, 50, 100, 200, 500]

A `MultiLabelBinarizer` is used to accomplish encoding of tags into binary arrays.

## Refinement

Based on the results of GridSearchCV, the optimal parameters for the pipeline classifier are:

  DT__estimator__criterion: 'gini'
	DT__estimator__max_depth: 25
	union__content__tfidf__max_df: 0.9
	union__content__tfidf__min_df: 0.001
	union__title__tfidf__max_df: 0.9
	union__title__tfidf__min_df: 0.001
	union__transformer_weights: {'content': 0.4, 'title': 0.6}

where `transformer_weights` refer to how much the FeatureUnion scales the Tfidf vectors when combining them and thus providing more weight to the title of a question.

The trees themselves are fairly shallow, at a max depth of 25, so overfitting the decisiontrees was taken into account by pruning.

These parameters are set and used to make predictions on the test data.


### Results

## Model Evaluation and Validation
The metrics discussed at the beginning of the investigation (the F1 score) can be applied in two scenarios to judge the health and validate the designed and cross validated classifier pipeline.

The first is the reported F1 score from the fitted GridSearchCV instance which checks predictions against input data as if it were to make those guesses when it was fitting.

When running this on only one of the 6 CSV files, the reported score is
which is a non-trivial result.

However, the big question of this investigation is whether this knowledge can be transferred to physics and unfortunately, the pipeline strongly underperforms on new features.
Since the nature of the challenge was a contest, the predictions created by the pipeline were submitted for evaluation online and received a measly score 0.8 %.

The most glaring problem with the resultant physics predictions is that almost 50% of the 81,926 questions (40,569 to be exact) are simply blank and the classifier did not predict anything.

## Justification

While the target problem is only marginally addressed, with a success rate under 1%, the classification system performed well on features within its respective topic set with scores in the mid 30%'s, which was the mean of the scores on Kaggle's leaderboard.

### Conclusion

## Free-Form Visualization

Seen below are 4 various decision trees constructed for specific labels after fitting on the training data.

The numbers indexed by the `X` are the TfidfVectorizer encodings and their resultant values. Ideally, the information to decode these to their original strings can be provided to the `tree.export_graphviz` function as a parameter using the `.get_features_names` method, however, since there was a custom transformer introduced (`Selector` class) to subset the input data by title and text and combine it using a FeatureUnion, the pipeline was unable to provide the list of feature_names according to the encodings. Attempting to patch the class through decorators and with other libraries (singledispatch and eli5) provided little help.

The gini values are a statistical measure of dispersion and are related through entropy--along these inequalities the tree formulates its decisions. The given examples analyze the distinctions between the tags `3d-model` and `3d-printing` according to the Tfidf values.

## Reflection

While it is certainly anticlimactic to receive such low results by the classifier when out in the wild, it is important to keep in mind that this challenge had an element of facetiousness inherently and it is presumptuous to think that there is a unifying thread to a slice of knowledge. Pragmatically, the issue all along is that as a text classifier, both the features and the tags share a different vocabulary and is very difficult to correlate.

## Improvement

With ample time, this experiment could be expanded by investigating the effects of scaling the classifier and seeing if there is a tipping point at which it loses accuracy. Overfitting is a usual suspect when using decision trees, however, this problem was mitigated by trying different combinations of depth with relatively standard input data sets. Scale comes into play again here and the question becomes if fitting the entire dataset at once plays well with the depth chosen at fitting.
Other classifiers are always fair contestants when trying to tackle larger datasets: k-nearest-neighbor (kNN) and Support Vector Machine (SVM) would be good candidates and would require their own parameter tunings.
