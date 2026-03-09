# chatbot
The Emotion Detection Chatbot is a Natural Language Processing (NLP) project that analyzes user text and predicts the emotion expressed in the message.
The Emotion Detection Chatbot is a Natural Language Processing (NLP) project that analyzes user messages and predicts the emotion expressed in the text. The chatbot uses Machine Learning algorithms trained on labeled textual data to classify emotions such as happiness, sadness, anger, fear, and surprise.

This project demonstrates the complete workflow of a machine learning application, including:

Text preprocessing

Feature extraction

Model training

Hyperparameter tuning

Emotion prediction using a chatbot interface

Emotion detection systems are useful in many real-world applications such as:

Customer feedback analysis

Mental health monitoring

Social media sentiment analysis

Human-computer interaction systems

Project Objectives

The main objectives of this project are:

Build an intelligent chatbot capable of detecting emotions from user text.

Apply Natural Language Processing techniques for text preprocessing.

Train and evaluate machine learning classification models.

Improve model performance using Hyperparameter Tuning.

Implement a chatbot that interacts with users and predicts emotions in real time.

📚 Dataset Description

The dataset contains text samples labeled with emotions. Each entry includes:

Column	Description
Text	A sentence expressing an emotion
Emotion	The label representing the emotion
Example
Text	Emotion
I feel amazing today	Joy
I am scared about the future	Fear
This situation makes me angry	Anger

The dataset allows machine learning models to learn patterns associated with emotional expressions in text.

🧹 Data Preprocessing

Raw text data must be cleaned before training the models.

The following preprocessing steps were applied:

1️⃣ Lowercasing

All text is converted to lowercase to ensure consistency.

Example:

"I Am Happy" → "i am happy"
2️⃣ Removing Punctuation

Special characters and punctuation are removed from the text.

Example:

"I am happy!!!" → "I am happy"
3️⃣ Removing Stopwords

Common words that do not carry significant meaning are removed.

Examples of stopwords:

the, is, at, on, in, and
4️⃣ Lemmatization

Lemmatization reduces words to their base form.

Example:

running → run
better → good

This step helps reduce vocabulary size and improves model performance.

🔢 Feature Extraction

After preprocessing, the text must be converted into numerical vectors so that machine learning algorithms can process it.

The project uses TF-IDF (Term Frequency – Inverse Document Frequency).

TF-IDF measures how important a word is in a document relative to the entire dataset.

Advantages:

Reduces the importance of common words

Highlights meaningful words

Improves classification performance

🧠 Machine Learning Models

Two machine learning algorithms were implemented and evaluated.

1️⃣ Naive Bayes (MultinomialNB)

Naive Bayes is a probabilistic classifier based on Bayes’ theorem. It assumes that features are independent of each other.

Advantages:

Fast training and prediction

Efficient for text classification

Works well with high-dimensional data

2️⃣ Decision Tree Classifier

Decision Trees classify data by learning decision rules derived from the features.

Advantages:

Easy to interpret

Handles complex patterns

Provides decision logic

⚙️ Hyperparameter Tuning

To improve the performance of the models, Hyperparameter Tuning was performed.

Hyperparameters are parameters set before training the model.

Examples:

Naive Bayes
alpha → smoothing parameter
Decision Tree
max_depth
min_samples_split
criterion (gini / entropy)

The project uses GridSearchCV to automatically test multiple parameter combinations and select the best configuration.

This process helps maximize model performance.

📊 Model Evaluation

The dataset was divided into:

80% Training Data

20% Testing Data

The primary evaluation metric used is Accuracy.

Model	Accuracy
Naive Bayes	91%
Decision Tree	86%

Results indicate that Naive Bayes performs slightly better for this text classification task.

