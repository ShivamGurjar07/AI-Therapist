#important libraries
import pandas as pd
import numpy as np
import nltk
import re
#importing stopwords is optional, in this case it decreased accuracy
#from nltk.corpus import stopwords
import itertools
import time


start_time = time.time()




data = pd.read_csv('text_emotion.csv',delimiter='\t')
#data = data.iloc[:100,:]


#stopset = set(stopwords.words('english'))

from nltk.stem.wordnet import WordNetLemmatizer 
lem = WordNetLemmatizer()

#comprehensive cleaning
def cleaning(text):
    txt = str(text)
    txt = re.sub(r"http\S+", "", txt)
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
            for k in range(len(txt)-1):
                words+= " " + txt[k+1]
            txt = words
            txt = re.sub(r'[^\w]', ' ', txt)
            if len(txt) == 0:
                return 'no text'
            else:
                txt = ''.join(''.join(s)[:2] for _, s in itertools.groupby(txt))
                txt = txt.replace("'", "")
                txt = nltk.tokenize.word_tokenize(txt)
                #data.content[i] = [w for w in data.content[i] if not w in stopset]
                for j in range(len(txt)):
                    txt[j] = lem.lemmatize(txt[j], "v")
                if len(txt) == 0:
                    return 'no text'
                else:
                    return txt
                
data['content'] = data['content'].map(lambda x: cleaning(x))


        
data = data.reset_index(drop=True)
for i in range(len(data)):
    words = data.content[i][0]
    for j in range(len(data.content[i])-1):
        words+= ' ' + data.content[i][j+1]
    data.content[i] = words

print(type(data.content[5]))
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(data.content, data.sentiment, test_size=0.25, random_state=0)

x_train = x_train.reset_index(drop = True)
x_test = x_test.reset_index(drop = True)

y_train = y_train.reset_index(drop = True)
y_test = y_test.reset_index(drop = True)
vectorizer = TfidfVectorizer(min_df=3, max_df=0.9)

train_vectors = vectorizer.fit_transform(x_train)
test_vectors = vectorizer.transform(x_test)

model = svm.SVC(kernel='linear') 
model.fit(train_vectors, y_train) 
predicted_sentiment = model.predict(test_vectors)

print(classification_report(y_test, predicted_sentiment))

predicted_sentiments = []
for s in range(len(predicted_sentiment)):
    predicted_sentiments.append(predicted_sentiment[s])
    
prediction_df = pd.DataFrame({'Content':x_test, 'Emotion_predicted':predicted_sentiment, 'Emotion_actual': y_test})
prediction_df.to_csv('emotion_recognizer_svm.csv', index = False)

elapsed_time = time.time() - start_time
print ("processing time:", elapsed_time, "seconds")




import random
import webbrowser
while(1):
        print("**hi,i am your personal Therapist,write something here.....")
        s=input()
        s=s.lower()
        #print(s)
        
        s=cleaning(s)
        words = s[0]
        for j in range(len(s)-1):
            words+= ' ' + s[j+1]
        s = words

        s=vectorizer.transform([s])
        print(model.predict(s)
        x=model.predict(s)
        if(x=='sadness'):
            print(random.choice(open("sad1.txt").readlines()))
            s=input()
            s=s.lower()
            print(random.choice(open("sad2.txt").readlines()))
            s=input()
            s=s.lower()
            #print(s)
            if(s in {"okay","ok","yes","fine","show","show me","yup","go ahead","what"}):
                str1=random.choice(open("sad3.txt").readlines())
                for i in range(0,len(str1)):
                    if(str1[i:i+3]=='web'):
                        print(str1[0:i])
                        url=str1[i+3:]
                        webbrowser.open(url)
                        break
    
                else:
                    print(str1)
                
            else:
                print('would you like a medical checkup')
                s=input()
                s=s.lower()
                if(s in {"okay","ok","yes","fine","show","show me","yup","go ahead","what"}):
                    runfile("bott.py")
                else:
                    continue
        elif(x=='joy'):
            print(random.choice(open("joy1.txt").readlines()))
            s=input()
            s=s.lower()
            print(random.choice(open("joy2.txt").readlines()))
            s=input()
            s=s.lower()
            if(s in {"yes please","okay","ok","yes","fine","show","show me","yup","go ahead","what"}):
                str1=random.choice(open("joy3.txt").readlines())
                for i in range(0,len(str1)):
                    if(str1[i:i+3]=='web'):
                        print(str1[0:i])
                        url=str1[i+3:]
                        webbrowser.open(url)
                        break
    
                else:
                    print(str1)
                
            else:
                continue
        elif(x=='fear'):
            print(random.choice(open("fear1.txt").readlines()))
            s=input()
            s=s.lower()
            print(random.choice(open("fear2.txt").readlines()))
            s=input()
            s=s.lower()
            if(s in {"okay","ok","yes","fine","show","show me","yup","go ahead","what"}):
                str1=random.choice(open("sad3.txt").readlines())
                for i in range(0,len(str1)):
                    if(str1[i:i+3]=='web'):
                        print(str1[0:i])
                        url=str1[i+3:]
                        webbrowser.open(url)
                        break
    
                else:
                    print(str1)
                
            else:
                print('would you like a medical checkup')
                s=input()
                s=s.lower()
                if(s in {"okay","ok","yes","fine","show","show me","yup","go ahead","what"}):
                    runfile("bott.py")
                else:
                    continue
        elif(x=='anger'):
            print(random.choice(open("anger1.txt").readlines()))
            s=input()
            s=s.lower()
            print(random.choice(open("anger2.txt").readlines()))
            s=input()
            s=s.lower()
            if(s in {"yes please","okay","ok","yes","fine","show","show me","yup","go ahead","what"}):
                str1=random.choice(open("anger3.txt").readlines())
                for i in range(0,len(str1)):
                    if(str1[i:i+3]=='web'):
                        print(str1[0:i])
                        url=str1[i+3:]
                        webbrowser.open(url)
                        break
    
                else:
                    print(str1)
                
            else:
                continue