from preprocessing import preprocess_data
from embeddings import generate_embeddings
from training import train_model
import os

def run_pipeline(input_csv, output_dir='outputs'):
    print("="*60)
    print("PIPELINE ML NLP - DÉMARRAGE")
    print("="*60)
    
    # Créer dossiers
    os.makedirs(f"{output_dir}", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Chemins
    cleaned_csv = f"{output_dir}/cleaned.csv"
    embeddings_npy = f"{output_dir}/embeddings.npy"
    data_pkl = f"{output_dir}/data_with_emb.pkl"
    model_file = "models/model.joblib"
    encoder_file = "models/encoder.joblib"
    
    # ÉTAPE 1: Preprocessing
    print("\n" + "="*60)
    print("ÉTAPE 1/3: PREPROCESSING")
    print("="*60)
    preprocess_data(input_csv, cleaned_csv)
    
    # ÉTAPE 2: Embeddings
    print("\n" + "="*60)
    print("ÉTAPE 2/3: EMBEDDINGS")
    print("="*60)
    generate_embeddings(cleaned_csv, embeddings_npy, data_pkl)
    
    # ÉTAPE 3: Training
    print("\n" + "="*60)
    print("ÉTAPE 3/3: TRAINING")
    print("="*60)
    train_model(data_pkl, embeddings_npy, model_file, encoder_file)
    
    print("\n" + "="*60)
    print("PIPELINE TERMINÉ AVEC SUCCÈS!")
    print("="*60)

if __name__ == "__main__":
    import sys
    input_file = sys.argv[1] if len(sys.argv) > 1 else 'data/raw/dataset.csv'
    run_pipeline(input_file)