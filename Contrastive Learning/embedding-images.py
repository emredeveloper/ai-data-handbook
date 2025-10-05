import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import numpy as np
import random
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt

# --- 1. Dataset Preparation ---
# We need to create pairs of images for the training.
# A pair will consist of two images and a label (1 if same class, 0 if different).

class SiameseMNIST(Dataset):
    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset
        self.labels = mnist_dataset.targets
        self.data = mnist_dataset.data
        self.labels_set = set(self.labels.numpy())
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, index):
        # Get a random image (anchor)
        img1, label1 = self.data[index], self.labels[index].item()

        # Decide if this will be a positive or negative pair
        should_get_positive = random.randint(0, 1)
        if should_get_positive:
            # Get another image from the same class
            positive_index = index
            while positive_index == index:
                positive_index = random.choice(self.label_to_indices[label1])
            img2, label2 = self.data[positive_index], self.labels[positive_index].item()
            target = torch.tensor(1, dtype=torch.float)
        else:
            # Get an image from a different class
            negative_label = random.choice(list(self.labels_set - {label1}))
            negative_index = random.choice(self.label_to_indices[negative_label])
            img2, label2 = self.data[negative_index], self.labels[negative_index].item()
            target = torch.tensor(0, dtype=torch.float)

        # Add a channel dimension and normalize
        img1 = img1.unsqueeze(0).float() / 255.0
        img2 = img2.unsqueeze(0).float() / 255.0
        
        return img1, img2, target

# --- 2. Model Architecture ---
# A simple CNN that will act as the "tower" in our siamese network.
class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(1, 32, 5), nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, 5), nn.PReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 256), nn.PReLU(),
            nn.Linear(256, 256), nn.PReLU(),
            nn.Linear(256, 2) # The final embedding dimension is 2
        )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

# --- 3. Contrastive Loss Function ---
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        
        loss_contrastive = torch.mean(
            (label) * torch.pow(euclidean_distance, 2) +
            (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss_contrastive

# --- 4. Training ---
def train():
    # Hyperparameters
    epochs = 5
    lr = 0.001
    batch_size = 64
    margin = 1.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    mnist_train = MNIST(root='./data', train=True, download=True, transform=ToTensor())
    siamese_train_dataset = SiameseMNIST(mnist_train)
    train_loader = DataLoader(siamese_train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss, and optimizer
    embedding_net = EmbeddingNet().to(device)
    model = SiameseNet(embedding_net).to(device)
    criterion = ContrastiveLoss(margin=margin)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (img1, img2, label) in enumerate(train_loader, 1):
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)

            optimizer.zero_grad()
            output1, output2 = model(img1, img2)
            loss = criterion(output1, output2, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Step {i}/{len(train_loader)}, Loss: {running_loss/100:.4f}")
                running_loss = 0.0
    
    print("Finished Training")
    return model.embedding_net

# --- 5. Visualization ---
def visualize_embeddings(embedding_net, device):
    mnist_test = MNIST(root='./data', train=False, download=True, transform=ToTensor())
    test_loader = DataLoader(mnist_test, batch_size=1000, shuffle=False)

    embedding_net.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            output = embedding_net(images)
            embeddings.append(output.cpu().numpy())
            labels.append(targets.cpu().numpy())

    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)

    # Use t-SNE if embeddings are high-dimensional, otherwise plot directly
    if embeddings.shape[1] > 2:
        print("Running t-SNE...")
        tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
        embeddings_2d = tsne.fit_transform(embeddings)
    else:
        embeddings_2d = embeddings
        
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='jet', alpha=0.6)
    plt.legend(handles=scatter.legend_elements()[0], labels=list(range(10)))
    plt.title("MNIST Embeddings Visualized")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.tight_layout()
    plt.savefig("mnist_embeddings.png", dpi=200)
    plt.show()

# --- 6. Evaluation ---
def evaluate_embeddings(embedding_net, device, num_pairs=10000):
    mnist_test = MNIST(root='./data', train=False, download=True, transform=ToTensor())
    data = mnist_test.data
    targets = mnist_test.targets

    labels_set = set(targets.numpy())
    label_to_indices = {label: np.where(targets.numpy() == label)[0] for label in labels_set}

    embedding_net.eval()
    y_true = []
    distances = []

    half = num_pairs // 2
    with torch.no_grad():
        # Positive pairs
        for _ in range(half):
            label = random.choice(list(labels_set))
            idx1, idx2 = np.random.choice(label_to_indices[label], size=2, replace=False)
            img1 = (data[idx1].unsqueeze(0).float() / 255.0).to(device)
            img2 = (data[idx2].unsqueeze(0).float() / 255.0).to(device)
            emb1 = embedding_net(img1.unsqueeze(0))
            emb2 = embedding_net(img2.unsqueeze(0))
            dist = F.pairwise_distance(emb1, emb2, keepdim=False).item()
            distances.append(dist)
            y_true.append(1)

        # Negative pairs
        for _ in range(num_pairs - half):
            label1, label2 = random.sample(list(labels_set), 2)
            idx1 = random.choice(label_to_indices[label1])
            idx2 = random.choice(label_to_indices[label2])
            img1 = (data[idx1].unsqueeze(0).float() / 255.0).to(device)
            img2 = (data[idx2].unsqueeze(0).float() / 255.0).to(device)
            emb1 = embedding_net(img1.unsqueeze(0))
            emb2 = embedding_net(img2.unsqueeze(0))
            dist = F.pairwise_distance(emb1, emb2, keepdim=False).item()
            distances.append(dist)
            y_true.append(0)

    distances = np.array(distances)
    y_true = np.array(y_true)

    # Lower distance should indicate same-class (label=1)
    # Convert distances to similarity scores for AUC: score = -distance
    try:
        auc = roc_auc_score(y_true, -distances)
    except Exception:
        auc = float('nan')

    # Find best threshold by maximizing accuracy
    candidate_thresholds = np.quantile(distances, np.linspace(0.01, 0.99, 99))
    best_acc = 0.0
    best_th = candidate_thresholds[len(candidate_thresholds)//2] if len(candidate_thresholds) else 0.5
    for th in candidate_thresholds:
        y_pred = (distances <= th).astype(int)
        acc = accuracy_score(y_true, y_pred)
        if acc > best_acc:
            best_acc = acc
            best_th = th

    y_pred = (distances <= best_th).astype(int)
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)

    pos_mean = float(distances[y_true == 1].mean()) if (y_true == 1).any() else float('nan')
    neg_mean = float(distances[y_true == 0].mean()) if (y_true == 0).any() else float('nan')

    print(f"Eval - Pairs: {num_pairs}")
    print(f"Eval - AUC: {auc:.4f}")
    print(f"Eval - Best threshold: {best_th:.4f}")
    print(f"Eval - Accuracy: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    print(f"Eval - Mean distance (same): {pos_mean:.4f} | (diff): {neg_mean:.4f}")

    return {
        'auc': auc,
        'best_threshold': float(best_th),
        'accuracy': float(acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'pos_mean_distance': pos_mean,
        'neg_mean_distance': neg_mean,
    }

# --- Run everything ---
if __name__ == '__main__':
    trained_embedding_net = train()
    evaluate_embeddings(trained_embedding_net, torch.device("cuda" if torch.cuda.is_available() else "cpu"), num_pairs=5000)
    visualize_embeddings(trained_embedding_net, torch.device("cuda" if torch.cuda.is_available() else "cpu"))