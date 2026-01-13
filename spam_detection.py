from django.shortcuts import render

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from.forms import MessageForm

dataset = pd.read_csv('C:/Users/DELL/OneDrive/Desktop/CHATBOT/backend/email.csv')

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(dataset['label'])

X_train, X_test, y_train, y_test = train_test_split(X, dataset['label'], test_size=0.2)

model = MultinomialNB()
model.fit(X_train, y_train)

def predictMessage(message):
  messageVector = vectorizer.transform([message])
  prediction = model.predict(messageVector)
  return 'spam' if prediction[0] == 1 else 'Ham'

def Home(request):
  result = None
  if request.method == 'POST':
    form = MessageForm(request.POST)
    if form.is_valid():
      message = form.cleaned_data['text']
      result = predictMessage(message)

  else:
    form = MessageForm()

  return render(request, 'home.html', {'form': form, 'result': result})


