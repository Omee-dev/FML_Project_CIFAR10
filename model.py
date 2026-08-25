import torch
import os
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split
import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score

def create_helmert_matrix(num_classes=10):
    helmert_matrix = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        helmert_matrix[0][i] = 1.0 / math.sqrt(num_classes)
    for i in range(1, num_classes):
        sqri = 1.0 / math.sqrt(i * (i + 1))
        for j in range(i):
            helmert_matrix[i][j] = sqri
        helmert_matrix[i][i] -= i * sqri
    helmert_matrix = helmert_matrix[1:, :]
    return helmert_matrix

def one_hot_encode(labels, num_classes):
    one_hot = torch.zeros(labels.size(0), num_classes, device=labels.device)
    one_hot[torch.arange(labels.size(0)), labels] = 1.0
    return one_hot

def batch_label_smoothing(labels, num_classes, gamma=0.01):
    batch_size = labels.size(0)
    uniform = torch.full((batch_size, num_classes), 1.0 / num_classes, device=labels.device)
    one_hot = one_hot_encode(labels, num_classes)
    smoothed_labels = (1 - gamma) * one_hot + gamma * uniform
    return smoothed_labels

def ilr_transform_batch(compositions, helmert_matrix):
    geo_mean = compositions.prod(dim=1, keepdim=True) ** (1.0 / compositions.size(1))
    clr = torch.log(compositions) - torch.log(geo_mean)
    helmert_matrix = torch.from_numpy(helmert_matrix).float().to(compositions.device)
    ilr_data = torch.matmul(helmert_matrix, clr.unsqueeze(2)).squeeze(2)
    return ilr_data

def inverse_ilr_transform_batch(ilr_data, helmert_matrix):
    helmert_matrix = torch.from_numpy(helmert_matrix).float().to(ilr_data.device)
    clr = torch.matmul(helmert_matrix.T, ilr_data.T).T
    exp_clr = torch.exp(clr)
    composition = exp_clr / exp_clr.sum(dim=1, keepdim=True)
    return composition

def add_noise_to_dataset(dataset, noise_rate=0.4, noise_type='symmetric'):
    if not 0 <= noise_rate <= 1:
        raise ValueError("Noise rate must be between 0 and 1")
    
    if noise_type not in ['symmetric', 'asymmetric']:
        raise ValueError("Noise type must be either 'symmetric' or 'asymmetric'")
    
    if isinstance(dataset, torch.utils.data.Subset):
        original_dataset = dataset.dataset
        indices = dataset.indices
        noisy_labels = np.array(original_dataset.targets)[indices]
    else:
        noisy_labels = np.array(dataset.targets)
    num_classes = len(np.unique(noisy_labels))

    if noise_type == 'symmetric':
        for i in range(len(noisy_labels)):
            if np.random.rand() < noise_rate:
                noisy_labels[i] = np.random.choice(np.delete(np.arange(num_classes), noisy_labels[i]))
                
    elif noise_type == 'asymmetric':
        # Asymmetric noise: flip labels to similar classes based on CIFAR-10 class relationships
        class_map = {
            0: 2,    # airplane -> bird
            1: 9,    # automobile -> truck
            2: 0,    # bird -> airplane
            3: 5,    # cat -> dog
            4: 7,    # deer -> horse
            5: 3,    # dog -> cat
            6: 4,    # frog -> deer
            7: 4,    # horse -> deer
            8: 1,    # ship -> automobile
            9: 1     # truck -> automobile
        }
        
        for i in range(len(noisy_labels)):
            if np.random.rand() < noise_rate:
                original_label = noisy_labels[i]
                noisy_labels[i] = class_map[original_label]

    if isinstance(dataset, torch.utils.data.Subset):
        for idx, original_idx in enumerate(indices):
            original_dataset.targets[original_idx] = noisy_labels[idx]
    else:
        dataset.targets = noisy_labels.tolist()

    return dataset


def construct_covariance_and_inverse(r):
    batch_size, rank = r.shape
    I = torch.eye(rank).to(r.device).unsqueeze(0).expand(batch_size, -1, -1)
    r = r.unsqueeze(2)
    outer_r = torch.bmm(r, r.transpose(1, 2))
    S = outer_r + I
    r_dot_r = torch.bmm(r.transpose(1, 2), r)
    S_inv = I - outer_r / (1 + r_dot_r)
    log_det_S = torch.log(1 + r_dot_r.squeeze(-1).squeeze(-1))
    return S_inv, log_det_S

class LabelCorrectionEMA:
    def __init__(self, model, device, decay=0.999):
        self.device = device
        self.model = model.to(device)
        self.decay = decay
        self.ema_model = self._clone_model()
        self.alpha = 0.99

    def _clone_model(self):
        ema_model = type(self.model)()
        ema_model.load_state_dict(self.model.state_dict())
        ema_model.to(self.device)
        for param in ema_model.parameters():
            param.detach_()
        return ema_model

    def update(self):
        for ema_param, model_param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data = self.decay * ema_param.data + (1.0 - self.decay) * model_param.data

    def compute_shift(self, epoch, inputs, ilr_labels, mu):
        transition_weight = 1 - (self.alpha ** epoch)
        
        with torch.no_grad():
            ema_mu, _ = self.ema_model(inputs)
        
        shift = ilr_labels - ema_mu
        corrected_mu = mu + transition_weight * shift
        
        return corrected_mu

def enhanced_multivariate_gaussian_loss(y_true, mu, r, shift=None):
    S_inv, log_det_S = construct_covariance_and_inverse(r)
    
    if shift is not None:
        mu = mu + shift
    
    diff = y_true - mu 
    mahalanobis_dist = torch.bmm(diff.unsqueeze(1), S_inv) 
    mahalanobis_dist = torch.bmm(mahalanobis_dist, diff.unsqueeze(2)).squeeze() 
    
    loss = 0.5 * mahalanobis_dist + 0.5 * log_det_S
    return loss.mean()

def prepare_data(noise_rate=0.4):
    global train_loader, val_loader, test_loader
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_dataset = add_noise_to_dataset(train_dataset, noise_rate=noise_rate, noise_type='symmetric')

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

class CIFAR10MultivariateModel(nn.Module):
    def __init__(self):
        super(CIFAR10MultivariateModel, self).__init__()
        self.feature_extractor = models.resnet18(pretrained=False)
        self.feature_extractor.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.feature_extractor.maxpool = nn.Identity() 
        self.feature_extractor.fc = nn.Identity() 

        self.fc_mu = nn.Linear(512, 9)
        self.fc_r = nn.Linear(512, 9)

    def forward(self, x):
        features = self.feature_extractor(x)
        mu = self.fc_mu(features)
        r = self.fc_r(features) 
        return mu, r

class MixupAugmentation:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        
    def __call__(self, x, y):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

def train_noise_corrected_model(device='cuda', epochs=60, noise_rate=0.4):
    global train_loader, val_loader, test_loader
    prepare_data(noise_rate=noise_rate)

    model = CIFAR10MultivariateModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    helmert_matrix = create_helmert_matrix(num_classes=10)

    label_correction_ema = LabelCorrectionEMA(model, device)

    train_accuracies, val_accuracies = [], []
    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0
        running_train_loss = 0.0

        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{epochs}', unit='batch') as pbar:
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                smoothed_one_hot_labels = batch_label_smoothing(labels, num_classes=10)
                ilr_labels = ilr_transform_batch(smoothed_one_hot_labels, helmert_matrix)

                optimizer.zero_grad()
                mu, r = model(inputs)

                if epoch > 20:
                    label_shift = label_correction_ema.compute_shift(epoch, inputs, ilr_labels, mu)
                    loss = enhanced_multivariate_gaussian_loss(ilr_labels, mu, r, shift=label_shift)
                else:
                    loss = enhanced_multivariate_gaussian_loss(ilr_labels, mu, r)
                
                loss.backward()
                optimizer.step()
                label_correction_ema.update()

                predicted_one_hot = inverse_ilr_transform_batch(mu, helmert_matrix)
                _, predicted = torch.max(predicted_one_hot, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                train_accuracy = 100. * correct / total
                
                running_train_loss += loss.item()
                pbar.set_postfix({'Train loss': loss.item(), 'Train Accuracy': train_accuracy})
                pbar.update(1)
        
        train_accuracies.append(train_accuracy)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                mu, r = model(inputs)
                predicted_one_hot = inverse_ilr_transform_batch(mu, helmert_matrix)
                _, predicted = torch.max(predicted_one_hot, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_accuracy = 100. * correct / total
        val_accuracies.append(val_accuracy)

        print(f"Epoch [{epoch+1}/{epochs}], Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}")
    # Load best model for testing
    model.eval()
    true_labels, model_preds = [], []
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            mu, _ = model(inputs)
            predicted_one_hot = inverse_ilr_transform_batch(mu, helmert_matrix)
            _, model_pred = torch.max(predicted_one_hot, 1)
            test_total += labels.size(0)
            test_correct += (model_pred == labels).sum().item()
            true_labels.extend(labels.cpu().numpy())
            model_preds.extend(model_pred.cpu().numpy())
    model_test_accuracy = 100. * test_correct / test_total
    model_f1 = f1_score(true_labels, model_preds, average='macro')
    model_conf_matrix = confusion_matrix(true_labels, model_preds)
    return train_accuracies, val_accuracies, model_f1, model_test_accuracy, model_conf_matrix

def train_baseline_model(device='cuda', epochs=60, noise_rate=0.4, learning_rate=1e-3):
    global train_loader, val_loader, test_loader

    baseline_model = models.resnet18(pretrained=False)
    baseline_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    baseline_model.maxpool = nn.Identity()
    baseline_model.fc = nn.Linear(baseline_model.fc.in_features, 10)
    baseline_model = baseline_model.to(device)

    optimizer = torch.optim.Adam(baseline_model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    train_accuracies, val_accuracies = [], []

    for epoch in range(epochs):
        baseline_model.train()
        correct, total = 0, 0
        running_train_loss = 0.0

        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{epochs}', unit='batch') as pbar:
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = baseline_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                train_accuracy = 100. * correct / total
                
                running_train_loss += loss.item()
                pbar.set_postfix({'Train loss': loss.item(), 'Train Accuracy': train_accuracy})
                pbar.update(1)
        
        train_accuracies.append(train_accuracy)

        baseline_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = baseline_model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_accuracy = 100. * correct / total
        val_accuracies.append(val_accuracy)

        print(f"Epoch [{epoch+1}/{epochs}], Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}")
    
    baseline_model.eval()
    true_labels, model_preds = [], []
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = baseline_model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
            true_labels.extend(labels.cpu().numpy())
            model_preds.extend(predicted.cpu().numpy())

    baseline_test_accuracy = 100. * test_correct / test_total
    baseline_f1 = f1_score(true_labels, model_preds, average='macro')
    baseline_conf_matrix = confusion_matrix(true_labels, model_preds)

    return train_accuracies, val_accuracies, baseline_f1, baseline_test_accuracy, baseline_conf_matrix

def plot_noise_rate_comparison(results, output_folder="plots"):
    os.makedirs(output_folder, exist_ok=True)
    
    for noise_rate, data in results.items():
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(data['custom_train_acc'], label='Custom Model')
        plt.plot(data['baseline_train_acc'], label='Baseline Model', linestyle='--')
        plt.title(f'Training Accuracy (Noise {noise_rate})')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(data['custom_val_acc'], label='Custom Model')
        plt.plot(data['baseline_val_acc'], label='Baseline Model', linestyle='--')
        plt.title(f'Validation Accuracy (Noise {noise_rate})')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plot_path = os.path.join(output_folder, f"accuracy_plots_noise_{noise_rate}.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"Plot saved: {plot_path}")
    
    plt.figure(figsize=(8, 5))
    custom_final_acc = [data['custom_val_acc'][-1] for data in results.values()]
    baseline_final_acc = [data['baseline_val_acc'][-1] for data in results.values()]
    noise_rates = list(results.keys())
    
    plt.plot(noise_rates, custom_final_acc, marker='o', label='Custom Model')
    plt.plot(noise_rates, baseline_final_acc, marker='x', label='Baseline Model')
    plt.title('Final Validation Accuracy vs Noise Rate')
    plt.xlabel('Noise Rate')
    plt.ylabel('Final Validation Accuracy')
    plt.legend()
    
    combined_plot_path = os.path.join(output_folder, "final_validation_accuracy_comparison.png")
    plt.tight_layout()
    plt.savefig(combined_plot_path)
    plt.close()
    print(f"Combined plot saved: {combined_plot_path}")

def run_experiments(device, noise_rates=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]):
    results = {}
    performance_summary = {}
    
    for noise_rate in noise_rates:
        print(f"\n--- Experiment with Noise Rate: {noise_rate} ---")
        
        custom_train_acc, custom_val_acc, custom_f1, custom_test_acc, custom_conf_matrix = train_noise_corrected_model(
            device=device, 
            epochs=60, 
            noise_rate=noise_rate
        )
        
        baseline_train_acc, baseline_val_acc, baseline_f1, baseline_test_acc, baseline_conf_matrix = train_baseline_model(
            device=device, 
            epochs=60, 
            noise_rate=noise_rate
        )
        
        results[noise_rate] = {
            'custom_train_acc': custom_train_acc,
            'custom_val_acc': custom_val_acc,
            'baseline_train_acc': baseline_train_acc,
            'baseline_val_acc': baseline_val_acc,
            'custom_conf_matrix': custom_conf_matrix,
            'baseline_conf_matrix': baseline_conf_matrix
        }
        
        performance_summary[noise_rate] = {
            'custom_f1': custom_f1,
            'custom_test_acc': custom_test_acc,
            'baseline_f1': baseline_f1,
            'baseline_test_acc': baseline_test_acc
        }
    
    plot_noise_rate_comparison(results)
    plot_confusion_matrices(results)
    save_performance_summary(performance_summary)
    

def plot_confusion_matrices(results, output_folder="plots"):
    os.makedirs(output_folder, exist_ok=True)
    
    noise_rates = list(results.keys())
    class_names = [str(i) for i in range(10)] 
    
    for noise_rate in noise_rates:
        plt.figure(figsize=(16, 6))
        
        plt.subplot(1, 2, 1)
        custom_cm = results[noise_rate]['custom_conf_matrix']
        sns.heatmap(custom_cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Custom Model Confusion Matrix\n(Noise Rate: {noise_rate})')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        plt.subplot(1, 2, 2)
        baseline_cm = results[noise_rate]['baseline_conf_matrix']
        sns.heatmap(baseline_cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Baseline Model Confusion Matrix\n(Noise Rate: {noise_rate})')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        plot_path = os.path.join(output_folder, f"confusion_matrix_noise_{noise_rate}.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"Confusion Matrix Plot saved: {plot_path}")

def save_performance_summary(performance_summary, output_folder="plots"):
    os.makedirs(output_folder, exist_ok=True)
    
    performance_path = os.path.join(output_folder, "model_performance_summary.csv")
    
    with open(performance_path, 'w') as f:
        f.write("Noise Rate,Custom Model F1,Custom Model Test Accuracy,Baseline Model F1,Baseline Model Test Accuracy\n")
        
        for noise_rate, metrics in performance_summary.items():
            f.write(f"{noise_rate},"
                    f"{metrics['custom_f1']:.4f},"
                    f"{metrics['custom_test_acc']:.4f},"
                    f"{metrics['baseline_f1']:.4f},"
                    f"{metrics['baseline_test_acc']:.4f}\n")
    
    print(f"Performance summary saved: {performance_path}")
    
    print("\n--- Performance Summary ---")
    for noise_rate, metrics in performance_summary.items():
        print(f"\nNoise Rate: {noise_rate}")
        print(f"Custom Model - F1 Score: {metrics['custom_f1']:.4f}, Test Accuracy: {metrics['custom_test_acc']:.4f}")
        print(f"Baseline Model - F1 Score: {metrics['baseline_f1']:.4f}, Test Accuracy: {metrics['baseline_test_acc']:.4f}")

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
run_experiments(device)