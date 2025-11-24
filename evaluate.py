import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, f1_score

# evaluate the model
def evaluate(model, loader, device, criterion):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    val_loss = 0

    with torch.inference_mode():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)

            loss = criterion(out, data.y)
            val_loss += loss.item() * data.num_graphs

            probs = F.softmax(out, dim=1)
            pred = out.argmax(dim=1)

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    val_loss = val_loss / len(loader.dataset)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    # Calculate AUC for multi-class
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
    except:
        auc = 0.0

    return accuracy, f1, auc, val_loss, all_labels, all_probs