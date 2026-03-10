import pandas as pd #pour lire et manipuler le dataset csv
import re #expressions régulières (nettoyage du texte)
import pickle #pour sauvegarder les modèles entraînés
import nltk  #bibliothèque nlp

from nltk.corpus import stopwords #mots inutiles (the, is, and…)
from nltk.stem import WordNetLemmatizer #réduit les mots à leur forme de base
from sklearn.model_selection import train_test_split  #diviser les données en entraînement et test
from sklearn.feature_extraction.text import TfidfVectorizer #transformer le texte en nombres.
from sklearn.naive_bayes import MultinomialNB #classifieur Naive Bayes
from sklearn.tree import DecisionTreeClassifier #classifieur arbre de décision.
from sklearn.metrics import accuracy_score #évaluer la performance.

# NLTK downloads
nltk.download("stopwords")
nltk.download("wordnet")

# Load dataset
df = pd.read_csv("data/emotions.csv")

lemmatizer = WordNetLemmatizer() #Initialise le lemmatizer.
stop_words = set(stopwords.words("english")) #Charge la liste des stopwords en anglais

# Preprocessing
def preprocess(text):
    text = text.lower() #convertit le texte en minuscules
    text = re.sub(r"[^a-z\s]", "", text) # supprimer les chiffres , ponctuation , symboles
    tokens = text.split() #découpe la phrase en mots.
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]#supprime les stopwords et applique la lemmatisation.
    return " ".join(tokens)#recompose la phrase nettoyée

df["text"] = df["text"].apply(preprocess) #Applique la fonction NLP à toutes les phrases du dataset.

X = df["text"]
y = df["label"]

# Vectorization
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# ===== Naive Bayes Model =====
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)
print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))

# Save Naive Bayes
pickle.dump(nb_model, open("model/nb_model.pkl", "wb"))

# ===== Decision Tree Model =====

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
pickle.dump(dt_model, open("model/dt_model.pkl", "wb"))

# Save Decision Tree
pickle.dump(dt_model, open("model/dt_model.pkl", "wb"))

# Save vectorizer
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))