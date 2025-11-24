import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from evaluate import evaluate
from datetime import datetime

#train for one epoch
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()

    all_preds = []
    all_labels = []
    all_probs = []

    total_loss = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

        probs = F.softmax(out, dim=1)
        pred = out.argmax(dim=1)

        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(data.y.cpu().numpy())
        all_probs.extend(probs.cpu().detach().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    # Calculate AUC for multi-class
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
    except:
        auc = 0.0

    total_loss = total_loss / len(loader.dataset)

    return total_loss, accuracy, f1, auc

#train the model
def train_model(model, train_loader, val_loader, optimizer, criterion, device,
                num_epochs, model_name, num_classes):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    train_f1_scores = []
    val_f1_scores = []
    train_auc_scores = []
    val_auc_scores = []

    best_val_acc = 0

    print(f"\nTraining {model_name}...")
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc, train_f1, train_auc = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_acc, val_f1, val_auc, val_loss, _, _ = evaluate(model=model, loader=val_loader, device=device, criterion=criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        train_f1_scores.append(train_f1)
        val_f1_scores.append(val_f1)
        train_auc_scores.append(train_auc)
        val_auc_scores.append(val_auc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            #save the best model
            #create models folder if it doesn't exist
            if not os.path.exists('models'):
                os.makedirs('models')
            torch.save(model.state_dict(), f"models/best_{model_name}_model.pth")

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] | timestamp: {datetime.now().strftime('%m-%d-%Y_%H:%M:%S')}")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train accuracy: {train_acc:.4f} | Val accuracy: {val_acc:.4f} \n Train F1: {train_f1:.4f} | Val f1: {val_f1:.4f} | Train auc: {train_auc:.4f} | Val auc: {val_auc:.4f} \n")

    return train_losses, val_losses, train_accuracies, val_accuracies, train_f1_scores, val_f1_scores, train_auc_scores, val_auc_scores