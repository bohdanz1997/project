import pickle
import os
import sys

from data_loader import DataLoader
from text_preprocessor import TextPreprocessor
from vectorizer_manager import VectorizerManager
from hybrid_classifier import HybridClassifier
from model_evaluator import ModelEvaluator
import config

def main():
    print("Loading data...")
    data_path = os.path.join(config.RAW_DATA_DIR, 'reviews.csv')
    loader = DataLoader(data_path)
    loader.load()
    
    print("Splitting data...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = loader.split_data()
    
    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")
    
    print("Preprocessing text...")
    preprocessor = TextPreprocessor()
    X_train_tokens = preprocessor.process_batch(X_train)
    X_val_tokens = preprocessor.process_batch(X_val)
    X_test_tokens = preprocessor.process_batch(X_test)
    
    print("Vectorizing...")
    vectorizer = VectorizerManager()
    X_train_vec = vectorizer.fit_transform(X_train_tokens)
    X_val_vec = vectorizer.transform(X_val_tokens)
    X_test_vec = vectorizer.transform(X_test_tokens)
    
    print(f"Feature dimension: {vectorizer.get_vocabulary_size()}")
    
    print("Training hybrid model...")
    model = HybridClassifier()
    model.train(X_train_vec, y_train)
    
    print("Evaluating on validation set...")
    evaluator = ModelEvaluator()
    y_val_pred = model.predict(X_val_vec)
    evaluator.evaluate(y_val, y_val_pred)
    evaluator.display_results()
    
    print("\nEvaluating on test set...")
    y_test_pred = model.predict(X_test_vec)
    evaluator.evaluate(y_test, y_test_pred)
    evaluator.display_results()
    
    print("\nSaving models...")
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    
    with open(os.path.join(config.MODELS_DIR, 'preprocessor.pkl'), 'wb') as f:
        pickle.dump(preprocessor, f)
    
    with open(os.path.join(config.MODELS_DIR, 'vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
    
    with open(os.path.join(config.MODELS_DIR, 'model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    
    print("Training completed!")

if __name__ == "__main__":
    main()
