import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

def train_model(data_pkl, embeddings_npy, model_output, encoder_output):
    print("Chargement des données...")
    with open(data_pkl, 'rb') as f:
        data = pickle.load(f)
    embeddings = np.load(embeddings_npy)
    
    print(f"Données: {len(data)}, Embeddings: {embeddings.shape}")
    
    # Préparation
    le = LabelEncoder()
    y = le.fit_transform(data['type'])
    X = embeddings
    
    print(f"Classes: {le.classes_}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Entraînement
    print("Entraînement du modèle SVM...")
    model = SVC(kernel='rbf', C=1.0, class_weight='balanced', 
                random_state=42, probability=True)
    model.fit(X_train, y_train)
    
    # Évaluation
    print("Évaluation...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nRapport de classification:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Sauvegarde
    joblib.dump(model, model_output)
    joblib.dump(le, encoder_output)
    print(f"\nModèle sauvegardé: {model_output}")
    print(f"Encodeur sauvegardé: {encoder_output}")
    
    return model, le

