import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

def clean_text(text, lang='en'):
    if pd.isna(text) or text == '':
        return []
    
    # Lowercase
    text = text.lower()
    
    # Suppression ponctuation
    text = re.sub(r'[^a-zàâçéèêëïîôùûüÿæœäöüß\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenisation
    lang_map = {'en': 'english', 'de': 'german'}
    try:
        tokens = word_tokenize(text, language=lang_map.get(lang, 'english'))
    except:
        tokens = text.split()
    
    # Suppression stopwords
    stop_words_dict = {
        'en': set(stopwords.words('english')),
        'de': set(stopwords.words('german'))
    }
    stop_words = stop_words_dict.get(lang, stop_words_dict['en'])
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    
    return tokens

def preprocess_data(input_csv, output_csv):
    print(f"Chargement: {input_csv}")
    df = pd.read_csv(input_csv)
    
    # Fusion subject + body si nécessaire
    if 'subject' in df.columns and 'body' in df.columns:
        df['email'] = df['subject'].fillna('') + ' ' + df['body'].fillna('')
    
    # Nettoyage
    print("Nettoyage des textes...")
    df['email_cleaned'] = df.apply(
        lambda row: clean_text(row['email'], row['language']), 
        axis=1
    )
    
    # Sauvegarde
    df.to_csv(output_csv, index=False)
    print(f"Sauvegardé: {output_csv}")
    return df

if __name__ == "__main__":
    import sys
    input_file = sys.argv[1] if len(sys.argv) > 1 else 'data/raw/dataset.csv'
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'data/processed/cleaned.csv'
    preprocess_data(input_file, output_file)