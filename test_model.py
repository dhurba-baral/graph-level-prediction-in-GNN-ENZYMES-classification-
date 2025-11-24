import torch
from models import GCN_1Layer, GCN_2Layer, GAT_Model
from evaluate import evaluate

#hyperparameters
from data_loaders import num_features, num_classes

hidden_channels = 64  #number of hidden units

#device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



#test the trained model
def test_model(model_path, test_loader, model_type, criterion):
    # Initialize model
    if model_type == 'GCN1':
        model = GCN_1Layer(num_features=num_features, hidden_channels=hidden_channels, num_classes=num_classes)
    elif model_type == 'GCN2':
        model = GCN_2Layer(num_features=num_features, hidden_channels=hidden_channels, num_classes=num_classes)
    elif model_type == 'GAT':
        model = GAT_Model(num_features=num_features, hidden_channels=hidden_channels, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    
    # Evaluate
    accuracy, f1, auc, val_loss, labels, probs = evaluate(model, test_loader, device, criterion)
    
    print(f"\nTest Results for {model_type}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    
    return accuracy, f1, auc, labels, probs