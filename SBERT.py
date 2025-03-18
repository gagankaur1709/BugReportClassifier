import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, auc, roc_curve
from sentence_transformers import SentenceTransformer
import joblib

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')


########## 1. Data Preprocessing and Cleaning ##########
def remove_html(text):
    """Remove HTML tags using a regex."""
    html = re.compile(r'<.*?>')
    return html.sub(r'', text)


def remove_emoji(text):
    """Remove emojis using a regex pattern."""
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"  # enclosed characters
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def preprocess_text(text):
    """
    Preprocess text by removing stopwords, punctuation, special characters,
    and applying lemmatization and stemming.
    """
    # Convert to lowercase
    text = text.lower()

    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    custom_stopwords = {'bug', 'issue', 'error', 'fix'}  # Add domain-specific stopwords
    stop_words.update(custom_stopwords)
    text = ' '.join([word for word in text.split() if word not in stop_words])

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

    # Stemming
    stemmer = PorterStemmer()
    text = ' '.join([stemmer.stem(word) for word in text.split()])

    return text


########## 2. Feature Extraction ##########
def extract_features(data):
    """
    Extract features using SBERT embeddings, metadata encoding, and sentiment analysis.
    """
    # SBERT Embeddings (for Title + Body)
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    text_embeddings = sbert_model.encode(data['text'].tolist())

    # Metadata Encoding (One-Hot Encoding for categorical features)
    metadata_columns = ['State', 'Labels']  # Use relevant metadata columns
    if all(col in data.columns for col in metadata_columns):
        encoder = OneHotEncoder(sparse_output=False)  # Use sparse_output=False
        metadata_encoded = encoder.fit_transform(data[metadata_columns])
    else:
        raise KeyError(f"Metadata columns {metadata_columns} not found in the dataset.")

    # Sentiment Analysis (VADER for Comments)
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = data['Comments'].apply(lambda x: sia.polarity_scores(x)['compound'])

    # Combine all features
    features = np.hstack([text_embeddings, metadata_encoded, sentiment_scores.values.reshape(-1, 1)])
    return features


########## 3. Classification ##########
def train_classifier(X_train, y_train):
    """
    Train an SVM classifier with RBF kernel using GridSearchCV for hyperparameter tuning.
    """
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto', 0.1, 1],
        'kernel': ['rbf']
    }
    svm = SVC(probability=True)  # Enable probability for AUC calculation
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='roc_auc')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


########## 4. Training & Evaluation ##########
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using accuracy, precision, recall, F1-score, and AUC.
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for AUC

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')  # Suppress warning
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    # AUC
    # If labels are 0/1 only, this works directly.
    # If labels are something else, adjust pos_label accordingly.
    fpr, tpr, _ = roc_curve(y_test, y_pred, pos_label=1)
    aucc = auc(fpr, tpr)

    # auc = roc_auc_score(y_test, y_pred_proba)

    return accuracy, precision, recall, f1, aucc


########## Main Script ##########
if __name__ == "__main__":
    # Load dataset
    project = 'pytorch'
    path = f'/Users/gagan/Documents/Dataset/{project}.csv'
    pd_all = pd.read_csv(path).sample(frac=1, random_state=999)  # Shuffle

    # Merge Title and Body
    pd_all['text'] = pd_all.apply(
        lambda row: row['Title'] + ' ' + row['Body'] if pd.notna(row['Body']) else row['Title'],
        axis=1
    )

    # Preprocess text
    pd_all['text'] = pd_all['text'].apply(remove_html)
    pd_all['text'] = pd_all['text'].apply(remove_emoji)
    pd_all['text'] = pd_all['text'].apply(preprocess_text)

    # Extract features
    X = extract_features(pd_all)
    y = pd_all['class']  # Target column

    # Repeat the process 30 times
    REPEAT = 30
    accuracies, precisions, recalls, f1_scores, auc_scores = [], [], [], [], []

    for i in range(REPEAT):
        # Split data into train and test sets (70-30 split)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i, stratify=y)

        # Train classifier
        classifier = train_classifier(X_train, y_train)

        # Evaluate model
        accuracy, precision, recall, f1, auc = evaluate_model(classifier, X_test, y_test)
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        auc_scores.append(auc)


    # Calculate average metrics
    avg_accuracy = np.mean(accuracies)
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_f1 = np.mean(f1_scores)
    avg_auc = np.mean(auc_scores)

    print("\n=== Average Metrics ===")
    print(f"Number of repeats:     {REPEAT}")
    print(f"Average Accuracy:  {avg_accuracy:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall:    {avg_recall:.4f}")
    print(f"Average F1-Score:  {avg_f1:.4f}")
    print(f"Average AUC:       {avg_auc:.4f}")

    # Save the model
    joblib.dump(classifier, 'performance_bug_classifier.pkl')
    print("Model saved as 'performance_bug_classifier.pkl'")