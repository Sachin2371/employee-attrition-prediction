from model import train_model

if __name__ == '__main__':
    accuracy, model_info = train_model()
    print(f"Model trained successfully!")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Algorithm: {model_info['algorithm']}")
    print(f"Features used: {model_info['features_used']}")
    print(f"Training samples: {model_info['training_samples']}")