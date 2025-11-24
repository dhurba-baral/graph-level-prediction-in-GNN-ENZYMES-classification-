import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize

#plot auc roc curve
def plot_roc_curves(num_classes, model_name, labels, probs):

    plt.figure(figsize=(12, 10))

    # Binarize labels
    labels_bin = label_binarize(labels, classes=range(num_classes))

    # Plot ROC for each class
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], probs[:, i])
        auc = roc_auc_score(labels_bin[:, i], probs[:, i])
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {auc:.3f})', linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curves - {model_name} - Test Set', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    #if plots folder doesn't exist, then
    if not os.path.exists('plots'):
      os.makedirs('plots')

    plt.savefig(os.path.join(f'plots/roc_curve_{model_name}.png'),
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

#plot loss curves
def plot_loss(train_losses, val_losses, model_name):
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.gca().set_xlabel('Epoch')
    plt.gca().set_ylabel('Loss')
    plt.gca().set_title(f'Loss curve {model_name}')
    plt.legend()

    #if plots folder doesn't exist, then
    if not os.path.exists('plots'):
      os.makedirs('plots')
      
    plt.savefig(f'plots/loss_curve_{model_name}')
    plt.show()

#plot accuracy curves
def plot_accuracies(train_accuracies, val_accuracies, model_name):
    plt.figure()
    plt.plot(train_accuracies, label='Train Acc')
    plt.plot(val_accuracies, label='Val Acc')
    plt.gca().set_xlabel('Epoch')
    plt.gca().set_ylabel('Accuracy')
    plt.gca().set_title(f'Accuracy curve {model_name}')
    plt.legend()

    #if plots folder doesn't exist, then
    if not os.path.exists('plots'):
      os.makedirs('plots')

    plt.savefig(f'plots/accuracy_curve_{model_name}')
    plt.show()