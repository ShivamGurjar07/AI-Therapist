#important libraries
import webbrowser
import random
import pandas as pd
import numpy as np
import nltk
import re
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from textblob.classifiers import NaiveBayesClassifier as NBC
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier

start_time = time.time()
###############################################################
print("shivam")

data = pd.read_csv('text_emotion.csv', delimiter='\t')
data = shuffle(data)
data = data.iloc[:, :]
print(data)

lem = nltk.stem.wordnet.WordNetLemmatizer()
#################################################################
#comprehensive cleaning
def cleaning(text):
    txt = str(text)
    txt = re.sub(r"http\S+", " ", txt)
    if len(txt) == 0:
        return 'no text'
    else:
        txt = txt.split()
        index = 0
        for j in range(len(txt)):
            if txt[j][0] == '@':
                index = j
        txt = np.delete(txt, index)
        if len(txt) == 0:
            return 'no text'
        else:
            words = txt[0]
            for k in range(len(txt) - 1):
                words += " " + txt[k + 1]
            txt = words
            txt = re.sub(r'[^\w]', ' ', txt)
            if len(txt) == 0:
                return 'no text'
            else:
                txt = ''.join(''.join(s)[:2] for _, s in itertools.groupby(txt))
                txt = txt.replace("'", "")
                txt = nltk.tokenize.word_tokenize(txt)
                for j in range(len(txt)):
                    txt[j] = lem.lemmatize(txt[j], "v")
                txt = [word for word in txt if len(word) > 2]  # Remove short words
                txt = ' '.join(txt)
                return txt

#########################################################################################
data['content'] = data['content'].map(lambda x: cleaning(x))
data = data[data['content'] != 'no text']  # Remove rows with empty text after cleaning

data = data.reset_index(drop=True)

x = int(np.round(len(data) * 0.75))
train = data.iloc[:x, :].reset_index(drop=True)
test = data.iloc[x:, :].reset_index(drop=True)

# Naive Bayes Classifier
training_corpus = [(train.content[k], train.sentiment[k]) for k in range(len(train))]
test_corpus = [(test.content[l], test.sentiment[l]) for l in range(len(test))]

nb_model = NBC(training_corpus)
nb_accuracy = nb_model.accuracy(test_corpus)
print("Naive Bayes Classifier Accuracy:", nb_accuracy)
nb_predictions = [nb_model.classify(test.content[m]) for m in range(len(test))]

# SVM Classifier
svm_model = make_pipeline(TfidfVectorizer(), SVC(kernel='linear'))
svm_model.fit(train.content, train.sentiment)
svm_accuracy = svm_model.score(test.content, test.sentiment)
print("SVM Classifier Accuracy:", svm_accuracy)
svm_predictions = svm_model.predict(test.content)

# K-Nearest Neighbors (KNN) Classifier
knn_model = make_pipeline(TfidfVectorizer(), KNeighborsClassifier())
knn_model.fit(train.content, train.sentiment)
knn_accuracy = knn_model.score(test.content, test.sentiment)
print("KNN Classifier Accuracy:", knn_accuracy)
knn_predictions = knn_model.predict(test.content)

print("Naive Bayes Classifier Report:\n", classification_report(test.sentiment, nb_predictions))
print("SVM Classifier Report:\n", classification_report(test.sentiment, svm_predictions))
print("KNN Classifier Report:\n", classification_report(test.sentiment, knn_predictions))

best_accuracy = max(nb_accuracy, svm_accuracy, knn_accuracy)
if best_accuracy == nb_accuracy:
    best_model = nb_model
    print("Best model: Naive Bayes Classifier")
elif best_accuracy == svm_accuracy:
    best_model = svm_model
    print("Best model: SVM Classifier")
else:
    best_model = knn_model
    print("Best model: KNN Classifier")

predictions_df = pd.DataFrame({'Content': test.content,
                               'NB_Emotion_Predicted': nb_predictions,
                               'SVM_Emotion_Predicted': svm_predictions,
                               'KNN_Emotion_Predicted': knn_predictions,
                               'Emotion_Actual': test.sentiment})
predictions_df.to_csv('emotion_recognizer_comparison.csv', index=False)

elapsed_time = time.time() - start_time

import pickle
if best_accuracy == nb_accuracy:
    filename = 'naive_bayes_model.sav'
elif best_accuracy == svm_accuracy:
    filename = 'svm_model.sav'
else:
    filename = 'knn_model.sav'
pickle.dump(best_model, open(filename, 'wb'))
print("Best model saved as:", filename)

################################################################################
print("shivam")
filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
model = loaded_model


#################################
##############################################################################
while True:
    print("** Hi, I am your personal Therapist. Write something here...")
    s = input().lower()
    s = cleaning(s)

    words = s[0]
    for j in range(len(s) - 1):
        words += ' ' + s[j + 1]
        s = words

    prediction = model.classify(s)

    print("Predicted Emotion:", prediction)

    if prediction == 'sadness':
        print(random.choice(open("sad1.txt").readlines()))
        s = input().lower()
        print(random.choice(open("sad2.txt").readlines()))
        s = input().lower()
        if s in {"okay", "ok", "yes", "fine", "show", "show me", "yup", "go ahead", "what"}:
            str1 = random.choice(open("sad3.txt").readlines())
            for i in range(0, len(str1)):
                if str1[i:i + 3] == 'web':
                    print(str1[0:i])
                    url = str1[i + 3:]
                    webbrowser.open(url)
                    break
            else:
                print(str1)
        else:
            print('Would you like a medical checkup?')
            s = input().lower()
            if s in {"okay", "ok", "yes", "fine", "show", "show me", "yup", "go ahead", "what"}:
                runfile("bott.py")

    elif prediction == 'joy':
        print(random.choice(open("joy1.txt").readlines()))
        s = input().lower()
        print(random.choice(open("joy2.txt").readlines()))
        s = input().lower()
        if s in {"yes please", "okay", "ok", "yes", "fine", "show", "show me", "yup", "go ahead", "what"}:
            str1 = random.choice(open("joy3.txt").readlines())
            for i in range(0, len(str1)):
                if str1[i:i + 3] == 'web':
                    print(str1[0:i])
                    url = str1[i + 3:]
                    webbrowser.open(url)
                    break
            else:
                print(str1)

    elif prediction == 'fear':
        print(random.choice(open("fear1.txt").readlines()))
        s = input().lower()
        print(random.choice(open("fear2.txt").readlines()))
        s = input().lower()
        if s in {"okay", "ok", "yes", "fine", "show", "show me", "yup", "go ahead", "what"}:
            str1 = "You should watch https://www.youtube.com/watch?v=_JCLwaAScq8"
            for i in range(0, len(str1)):
                if str1[i:i + 3] == 'web':
                    print(str1[0:i])
                    url = str1[i + 3:]
                    webbrowser.open(url)
                    break
            else:
                print(str1)
        else:
            print('Would you like a medical checkup?')
            s = input().lower()
            if s in {"okay", "ok", "yes", "fine", "show", "show me", "yup", "go ahead", "what"}:
                runfile("bott.py")

    elif prediction == 'anger':
        print(random.choice(open("anger1.txt").readlines()))
        s = input().lower()
        print(random.choice(open("anger2.txt").readlines()))
        s = input().lower()
        if s in {"yes please", "okay", "ok", "yes", "fine", "show", "show me", "yup", "go ahead", "what"}:
            str1 = random.choice(open("anger3.txt").readlines())
            for i in range(0, len(str1)):
                if str1[i:i + 3] == 'web':
                    print(str1[0:i])
                    url = str1[i + 3:]
                    webbrowser.open(url)
                    break
            else:
                print(str1)
