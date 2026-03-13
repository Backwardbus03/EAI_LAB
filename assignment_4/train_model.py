import os
import json
import joblib
import nltk
from nltk.corpus import twitter_samples
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import emoji

# Ensure datasets are downloaded
nltk.download('twitter_samples', quiet=True)

def train_and_save_model():
    print("Loading NLTK Twitter dataset...")
    # Load positive and negative tweets
    pos_tweets = twitter_samples.strings('positive_tweets.json')
    neg_tweets = twitter_samples.strings('negative_tweets.json')

    # Prepare data
    tweets = pos_tweets + neg_tweets
    labels = [1] * len(pos_tweets) + [0] * len(neg_tweets)

    # 1. Build Emoji Lexicon based on co-occurrence in training data
    print("Building Emoji Lexicon...")
    emoji_counts = {} # {emoji: {'pos': 0, 'neg': 0}}
    
    for tweet, label in zip(tweets, labels):
        emojis_present = [c for c in tweet if c in emoji.EMOJI_DATA]
        for e in emojis_present:
            if e not in emoji_counts:
                emoji_counts[e] = {'pos': 0, 'neg': 0}
            if label == 1:
                emoji_counts[e]['pos'] += 1
            else:
                emoji_counts[e]['neg'] += 1

    # Calculate sentiment score for each emoji based on probability
    emoji_lexicon = {}
    for e, counts in emoji_counts.items():
        total = counts['pos'] + counts['neg']
        if total > 5: # Only consider emojis that appear multiple times for significance
            # Score between -1 (negative) and 1 (positive)
            score = (counts['pos'] - counts['neg']) / total
            emoji_lexicon[e] = score

    # Include some hardcoded ones if they don't appear in twitter dataset enough
    hardcoded = {'😍': 0.9, '😂': 0.5, '😠': -0.8, '😡': -0.9, '😢': -0.7, '🙄': -0.3}
    for e, s in hardcoded.items():
        if e not in emoji_lexicon:
            emoji_lexicon[e] = s

    # 2. Train Text Model using pipeline
    print("Training Text Sentiment Classifier...")
    # Remove emojis from text for pure text training
    text_only_tweets = [''.join(c for c in t if c not in emoji.EMOJI_DATA) for t in tweets]

    X_train, X_test, y_train, y_test = train_test_split(text_only_tweets, labels, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
        ('clf', LogisticRegression(random_state=42))
    ])

    pipeline.fit(X_train, y_train)

    # Evaluate
    print("\nModel Evaluation:")
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

    # 3. Save Artifacts
    os.makedirs('models', exist_ok=True)
    
    model_path = os.path.join('models', 'text_model.joblib')
    joblib.dump(pipeline, model_path)
    print(f"Saved text model to {model_path}")

    lexicon_path = os.path.join('models', 'emoji_lexicon.json')
    with open(lexicon_path, 'w', encoding='utf-8') as f:
        json.dump(emoji_lexicon, f, ensure_ascii=False, indent=2)
    print(f"Saved emoji lexicon to {lexicon_path} (Vocab size: {len(emoji_lexicon)})")

if __name__ == "__main__":
    train_and_save_model()
