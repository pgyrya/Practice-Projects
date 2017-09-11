# -*- coding: utf-8 -*-
import pandas as pd
# Dataset from - https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
df = pd.read_table('smsspamcollection/SMSSpamCollection',
                   sep='\t', 
                   header=None, 
                   names=['label', 'sms_message'])

# Output printing out first 5 columns
df.head()

df['label'] = df.label.map({'ham':0, 'spam':1})
print(df.shape)
df.head() # returns (rows, columns)


documents = ['Hello, how are you!',
             'Win money, win from home.',
             'Call me now.',
             'Hello, Call hello you tomorrow?']

lower_case_documents = []
for i in documents:
    lower_case_documents.append(i.lower())
print(lower_case_documents)


sans_punctuation_documents = []
import string

for i in lower_case_documents:
    sans_punctuation_documents.append(i.translate(str.maketrans('', '', string.punctuation)))
print(sans_punctuation_documents)


preprocessed_documents = []
for i in sans_punctuation_documents:
    preprocessed_documents.append(i.split(' '))
print(preprocessed_documents)


frequency_list = []
import pprint
from collections import Counter

for i in preprocessed_documents:
    frequency_counts = Counter(i)
    frequency_list.append(frequency_counts)
pprint.pprint(frequency_list)


from sklearn.feature_extraction.text import CountVectorizer
count_vector = CountVectorizer()

count_vector.fit(documents)
count_vector.get_feature_names()

doc_array = count_vector.transform(documents).toarray()
doc_array

frequency_matrix = pd.DataFrame(doc_array, 
                                columns = count_vector.get_feature_names())
frequency_matrix




from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df['sms_message'], 
                                                    df['label'], 
                                                    random_state=1)

print('Number of rows in the total set: {}'.format(df.shape[0]))
print('Number of rows in the training set: {}'.format(X_train.shape[0]))
print('Number of rows in the test set: {}'.format(X_test.shape[0]))
print(X_train[1])

# Instantiate the CountVectorizer method
count_vector = CountVectorizer()

# Fit the training data and then return the matrix
training_data = count_vector.fit_transform(X_train)

# Transform testing data and return the matrix. Note we are not fitting the testing data into the CountVectorizer()
testing_data = count_vector.transform(X_test)

#explore encoding
print (count_vector.get_feature_names())
print(X_train[1])
print (training_data[1:3])
import scipy
print (scipy.sparse.csr_matrix.nonzero(training_data[1])[1])

#setup multinomial Naive Bayes classifier
from sklearn.naive_bayes import MultinomialNB
naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, y_train)

#visualize Naive Bayes feature scores
print(len(naive_bayes.feature_log_prob_[1]))
print(len(naive_bayes.feature_log_prob_[1]))
print(naive_bayes.feature_log_prob_[:,3])
print(enumerate(naive_bayes.feature_log_prob_[1,:]))
print(naive_bayes._get_coef())

feature_names = count_vector.get_feature_names()
log_probabilities = naive_bayes.feature_log_prob_[1,:]
words_probabilities = [[feature_names[i], log_probabilities[i]] for i in list(range(len(feature_names)))]
print (words_probabilities[1000:1005] )
def getkey(item):
    return item[1]
words_probabilities.sort(key = getkey)
print (words_probabilities[-50:-1] )

short_list= list(range(len(naive_bayes.feature_log_prob_[1]))
print(short_list)

predictions = naive_bayes.predict(testing_data)
print (sum(predictions[:100]))


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('Accuracy score: ', format(accuracy_score(y_test, predictions)))
print('Precision score: ', format(precision_score(y_test, predictions)))
print('Recall score: ', format(recall_score(y_test, predictions)))
print('F1 score: ', format(f1_score(y_test, predictions)))