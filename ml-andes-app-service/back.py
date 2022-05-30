import re
import nltk

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
wordnet_lemmatizer = WordNetLemmatizer()
nltk.download('wordnet')

def clean_text(text):
    text = re.sub("\'", "", text) 
    text = re.sub("[^a-zA-Z]"," ",text)
    text = ' '.join(text.split()) 
    text = text.lower() 
    return text
    
def remove_stopwords(text):
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    return ' '.join(no_stopword_text)

# Definición de la función que tenga como parámetro texto y devuelva una lista de lemas
def lemmas(text):
    text = text.lower()
    words = text.split()
    lemas = [wordnet_lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemas)