import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def test_import_modules():
    """Test que les modules s'importent correctement"""
    try:
        import preprocessing
        import embeddings
        import training
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")

def test_preprocessing_clean_text():
    """Test la fonction de nettoyage de texte"""
    from preprocessing import clean_text
    
    text = "Hello World! This is a TEST."
    result = clean_text(text, 'en')
    
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(token, str) for token in result)

def test_preprocessing_empty_text():
    """Test avec texte vide"""
    from preprocessing import clean_text
    
    assert clean_text('', 'en') == []
    assert clean_text(None, 'en') == []

def test_model_exists():
    """Vérifier que le modèle existe"""
    import os
    
    # Adapter selon votre structure
    model_path = 'models/model.joblib'
    if os.path.exists(model_path):
        assert True
    else:
        pytest.skip(f"Model not found at {model_path}")

def test_embeddings_shape():
    """Vérifier la forme des embeddings si disponibles"""
    import os
    
    embeddings_path = 'data/processed/embeddings.npy'
    if os.path.exists(embeddings_path):
        embeddings = np.load(embeddings_path)
        assert len(embeddings.shape) == 2
        assert embeddings.shape[1] > 0  # Dimension > 0
    else:
        pytest.skip(f"Embeddings not found at {embeddings_path}")