import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ===============================
# 1️⃣ Nettoyage et normalisation
# ===============================

def nettoyer_texte(texte):
    """
    Nettoie et normalise le texte :
    - minuscules
    - suppression ponctuation
    - suppression espaces multiples
    """
    texte = texte.lower()
    texte = re.sub(r"[^a-z\s]", "", texte)
    texte = re.sub(r"\s+", " ", texte).strip()
    return texte


# ===============================
# 2️⃣ Corpus émotionnel (dataset emotions – version sémantique)
# ===============================

emotion_corpus = {
    "sadness": [
        "i feel sad",
        "i feel empty",
        "i feel depressed",
        "i feel lonely",
        "nothing makes me happy"
    ],
    "fear": [
        "i feel stressed",
        "i am anxious",
        "i am worried about exams",
        "i feel pressure",
        "i am scared"
    ],
    "anger": [
        "i am angry",
        "i feel frustrated",
        "i am mad",
        "this makes me furious"
    ],
    "joy": [
        "i feel happy",
        "i am excited",
        "i feel great",
        "i am joyful"
    ],
    "surprise": [
        "i am surprised",
        "this is unexpected",
        "i did not expect this"
    ],
    "neutral": [
        "i am okay",
        "nothing special",
        "just talking"
    ]
}

# Nettoyage du corpus avec compréhension de liste
emotion_corpus_clean = {
    emotion: [nettoyer_texte(p) for p in phrases]
    for emotion, phrases in emotion_corpus.items()
}


# ===============================
# 3️⃣ Chargement du modèle Transformer
# ===============================

model = SentenceTransformer("all-MiniLM-L6-v2")

# Embeddings du corpus émotionnel
emotion_embeddings = {
    emotion: model.encode(phrases)
    for emotion, phrases in emotion_corpus_clean.items()
}


# ===============================
# 4️⃣ Réponses fixes (NON RANDOM ❌)
# ===============================

emotion_responses = {
    "sadness": (
        "I'm really sorry you're feeling this way 😢 "
        "Would you like to talk about what's making you feel sad?"
    ),
    "fear": (
        "I'm sorry you're feeling stressed 😟 "
        "Pressure and anxiety can be overwhelming. "
        "What is worrying you the most?"
    ),
    "anger": (
        "I can sense a lot of frustration 😡 "
        "What happened to make you feel this way?"
    ),
    "joy": (
        "That's really nice to hear 😄 "
        "What made you feel happy today?"
    ),
    "surprise": (
        "That sounds unexpected 😮 "
        "How did you react?"
    ),
    "neutral": (
        "I see 🤔 "
        "Can you tell me more about how you feel?"
    )
}


# ===============================
# 5️⃣ Fonction principale du chatbot
# ===============================

def chatbot_response(user_text):
    """
    Analyse sémantique du message utilisateur et
    retourne :
    - emotion détectée
    - réponse empathique stable
    """
    user_text_clean = nettoyer_texte(user_text)
    user_embedding = model.encode([user_text_clean])

    best_emotion = "neutral"
    best_score = 0.0

    for emotion, embeddings in emotion_embeddings.items():
        similarities = cosine_similarity(user_embedding, embeddings)
        score = np.max(similarities)

        if score > best_score:
            best_score = score
            best_emotion = emotion

    response = emotion_responses[best_emotion]
    return best_emotion, response