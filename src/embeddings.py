import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import pickle

def generate_embeddings(input_csv, output_npy, output_pkl, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    print(f"Chargement du modèle: {model_name}")
    model = SentenceTransformer(model_name)
    
    print(f"Chargement des données: {input_csv}")
    df = pd.read_csv(input_csv)
    
    # Conversion tokens en texte
    texts = []
    for tokens in df['email_cleaned']:
        if isinstance(tokens, str):
            tokens = eval(tokens)  # Convertir string liste en liste
        texts.append(' '.join(tokens) if isinstance(tokens, list) else str(tokens))
    
    # Génération embeddings
    print(f"Génération embeddings pour {len(texts)} textes...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    
    # Normalisation L2
    embeddings = normalize(embeddings, norm='l2')
    print(f"Shape: {embeddings.shape}")
    
    # Sauvegarde
    np.save(output_npy, embeddings)
    print(f"Embeddings sauvegardés: {output_npy}")
    
    df['embeddings'] = list(embeddings)
    with open(output_pkl, 'wb') as f:
        pickle.dump(df, f)
    print(f"DataFrame sauvegardé: {output_pkl}")
    
    return df, embeddings

