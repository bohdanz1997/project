import pickle
import os
import sys
import config

def load_models():
    models_dir = config.MODELS_DIR
    
    with open(os.path.join(models_dir, 'preprocessor.pkl'), 'rb') as f:
        preprocessor = pickle.load(f)
    
    with open(os.path.join(models_dir, 'vectorizer.pkl'), 'rb') as f:
        vectorizer = pickle.load(f)
    
    with open(os.path.join(models_dir, 'model.pkl'), 'rb') as f:
        model = pickle.load(f)
    
    return preprocessor, vectorizer, model

def predict_sentiment(text, preprocessor, vectorizer, model):
    tokens = preprocessor.process(text)
    vector = vectorizer.transform([tokens])
    prediction = model.predict(vector)[0]
    probabilities = model.predict_proba(vector)[0]
    
    sentiment = config.CLASSES[prediction]
    confidence = probabilities[prediction]
    
    return sentiment, confidence, probabilities

def main():
    print("Loading models...")
    preprocessor, vectorizer, model = load_models()
    
    print("Models loaded successfully!\n")
    
    while True:
        print("\nEnter review text (or 'quit' to exit):")
        text = input("> ")
        
        if text.lower() == 'quit':
            break
        
        if not text.strip():
            continue
        
        sentiment, confidence, probs = predict_sentiment(
            text, preprocessor, vectorizer, model
        )
        
        print(f"\nPredicted sentiment: {sentiment}")
        print(f"Confidence: {confidence:.4f}")
        print("\nAll probabilities:")
        for cls, prob in zip(config.CLASSES, probs):
            print(f"  {cls}: {prob:.4f}")

if __name__ == "__main__":
    main()
