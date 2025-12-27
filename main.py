"""
Model training, validation and testing code
"""
import os
import numpy as np
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from SSSTFN_model import SSSTFN
from dataset_fusion import MultimodalDataset, transform

num_epochs = 20
learning_rate = 0.0001
batch_size = 16
val_split = 0.2
save_dir = 'results'
log_filename = os.path.join(save_dir, 'SSSTFN_log.txt')
os.makedirs(save_dir, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def print_and_log(text):
    with open(log_filename, 'a') as f:
        print(text)
        f.write(text + '\n')

def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'train_history.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, phase='Validation'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{phase} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(save_dir, f'{phase.lower()}_confusion_matrix.png'))
    plt.close()

def train_one_epoch(model, device, dataloader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    for images, time_series, labels in dataloader:
        images, time_series, labels = images.to(device), time_series.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images, time_series)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return running_loss / len(dataloader), accuracy

def validate(model, device, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, time_series, labels in dataloader:
            images, time_series, labels = images.to(device), time_series.to(device), labels.to(device)

            outputs = model(images, time_series)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return running_loss / len(dataloader), accuracy, all_labels, all_preds

def test(model, device, image_test_path, time_series_test_path, batch_size=16):
    model.eval()

    test_dataset = MultimodalDataset(image_test_path, time_series_test_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_labels = []
    all_preds = []
    inference_times = []

    with torch.no_grad():
        for images, time_series, labels in test_loader:
            images, time_series, labels = images.to(device), time_series.to(device), labels.to(device)

            start_time = time.time()
            outputs = model(images, time_series)
            end_time = time.time()

            inference_time = (end_time - start_time) * 1000  # 转换为毫秒
            inference_times.append(inference_time)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    avg_inference_time = np.mean(inference_times)
    return accuracy, all_labels, all_preds, avg_inference_time

def main():
    total_start_time = time.time()

    print_and_log("Loading dataset...")
    full_dataset = MultimodalDataset('dataset/MCSTF_RGB', 'dataset/dataset.csv', transform=transform)

    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print_and_log(f"Training samples: {len(train_dataset)}")
    print_and_log(f"Validation samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = SSSTFN(num_classes=7).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print_and_log(f'Model Parameters (M): {total_params / 1e6:.2f}M')

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    print_and_log("\nStarting training...")
    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        train_loss, train_acc = train_one_epoch(model, device, train_loader, optimizer, criterion)

        val_loss, val_acc, _, _ = validate(model, device, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        print_and_log(f'Epoch [{epoch + 1}/{num_epochs}], '
                      f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, '
                      f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}, '
                      f'Time: {epoch_duration:.2f} seconds')

    print_and_log(f"\nTraining completed!")

    plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies)

    _, _, y_true, y_pred = validate(model, device, val_loader, criterion)
    val_report = classification_report(y_true, y_pred)
    print_and_log("\nValidation Classification Report:")
    print_and_log(val_report)
    plot_confusion_matrix(y_true, y_pred, phase='Validation')

    torch.save(model.state_dict(), os.path.join(save_dir, 'final_model.pth'))
    print_and_log(f"Final model saved to {os.path.join(save_dir, 'final_model.pth')}")

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    print_and_log(f"\nTotal training time: {total_duration:.2f} seconds.")

    print_and_log("\n" + "=" * 60)
    print_and_log("=== Testing on test set with final model ===")
    print_and_log("=" * 60)

    test_image_path = 'dataset/MCSTF_RGB_test'
    test_time_series_path = 'dataset/dataset_test.csv'

    print_and_log("Using final model for testing...")

    test_acc, test_labels, test_preds, avg_inference_time = test(
        model, device, test_image_path, test_time_series_path
    )

    print_and_log(f"Test Accuracy: {test_acc:.4f}")
    print_and_log(f"Average Inference Time (ms): {avg_inference_time:.4f}")

    test_report = classification_report(test_labels, test_preds)
    print_and_log(f"\nTest Classification Report:")
    print_and_log(test_report)

    plot_confusion_matrix(test_labels, test_preds, phase='Test')

    metrics = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'test_accuracy': test_acc,
        'avg_inference_time': avg_inference_time
    }

    np.save(os.path.join(save_dir, 'training_metrics.npy'), metrics)
    print_and_log(f"\nTraining metrics saved to {os.path.join(save_dir, 'training_metrics.npy')}")

if __name__ == "__main__":
    main()
