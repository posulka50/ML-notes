# Autoencoders (Автокодувальники)

## Що це?

**Autoencoder** — це **neural network**, яка навчається стискати (encode) дані в низькорозмірне представлення, а потім відновлювати (decode) їх назад. Це **unsupervised** метод для dimensionality reduction та feature learning.

**Головна ідея:** навчити мережу відтворювати вхід на виході, змушуючи її пропустити дані через "вузьке горлечко" (bottleneck), що створює compressed representation.

## Навіщо потрібні?

- 🗜️ **Dimensionality reduction** — нелінійна альтернатива PCA
- 🎨 **Feature learning** — автоматичне виявлення корисних features
- 🖼️ **Image compression** — стиснення зображень
- 🔍 **Anomaly detection** — виявлення аномалій
- 🎭 **Denoising** — видалення шуму з даних
- 🧬 **Generative models** — створення нових зразків (VAE)
- 📊 **Visualization** — 2D/3D embedding для візуалізації

## Коли використовувати?

**Потрібно:**
- **Нелінійні структури** — складні залежності в даних
- **Unsupervised learning** — немає labels
- **Deep features** — потрібні abstract representations
- **Великі дані** — багато зразків для навчання NN
- **Гнучкість** — різні архітектури для різних задач
- **Anomaly detection** — reconstruction error як міра

**Не потрібно:**
- **Малі дані** (< 1000 зразків) → PCA, t-SNE
- **Лінійні дані** → PCA простіший
- **Швидкість** критична → PCA швидше
- **Інтерпретованість** → PCA компоненти зрозуміліші
- **Простота** важлива → traditional methods

---

## Архітектура

### Базова структура

```
Input → Encoder → Bottleneck → Decoder → Output
                     ↓
               (latent space)
               (compressed)
```

**Компоненти:**

1. **Encoder:** стискає вхід в latent representation
2. **Bottleneck (latent space):** низькорозмірне представлення
3. **Decoder:** відновлює вхід з latent representation

### Діаграма

```
Input (784D)
     ↓
  [Dense 128] ← Encoder
     ↓
  [Dense 64]
     ↓
  [Dense 32] ← Bottleneck (latent space)
     ↓
  [Dense 64] ← Decoder
     ↓
  [Dense 128]
     ↓
Output (784D)

Мета: Output ≈ Input
```

### Математично

**Encoder:**
$$\mathbf{z} = f_{enc}(\mathbf{x})$$

**Decoder:**
$$\hat{\mathbf{x}} = f_{dec}(\mathbf{z})$$

**Loss (reconstruction error):**
$$L = \|\mathbf{x} - \hat{\mathbf{x}}\|^2 = \|\mathbf{x} - f_{dec}(f_{enc}(\mathbf{x}))\|^2$$

---

## Простий приклад: Linear Autoencoder

### Код (PyTorch)

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Завантажити дані
digits = load_digits()
X = digits.data  # (1797, 64)
y = digits.target

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# To tensor
X_tensor = torch.FloatTensor(X_scaled)

# Split
X_train, X_test = train_test_split(X_tensor, test_size=0.2, random_state=42)

print(f"Train shape: {X_train.shape}")  # (1437, 64)
print(f"Test shape: {X_test.shape}")    # (360, 64)

# Простий Autoencoder
class LinearAutoencoder(nn.Module):
    def __init__(self, input_dim=64, latent_dim=2):
        super(LinearAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)  # Bottleneck
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )
    
    def forward(self, x):
        z = self.encoder(x)  # Encode
        x_reconstructed = self.decoder(z)  # Decode
        return x_reconstructed
    
    def encode(self, x):
        return self.encoder(x)

# Створити модель
model = LinearAutoencoder(input_dim=64, latent_dim=2)
print(model)

# Loss та optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Навчання
num_epochs = 50
train_losses = []

for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, X_train)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    train_losses.append(loss.item())
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Візуалізація loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Оцінка на test
with torch.no_grad():
    test_outputs = model(X_test)
    test_loss = criterion(test_outputs, X_test)
    print(f'\nTest Loss: {test_loss.item():.4f}')

# Encode в 2D
with torch.no_grad():
    z_train = model.encode(X_train).numpy()
    z_test = model.encode(X_test).numpy()

# Візуалізація latent space
plt.figure(figsize=(10, 7))
scatter = plt.scatter(z_train[:, 0], z_train[:, 1], 
                     c=y[:len(z_train)], cmap='tab10', s=20, alpha=0.6)
plt.colorbar(scatter, label='Digit')
plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
plt.title('2D Latent Space (Autoencoder)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Порівняння оригінальних та відновлених зображень
n_display = 10
fig, axes = plt.subplots(2, n_display, figsize=(15, 3))

with torch.no_grad():
    reconstructed = model(X_test[:n_display]).numpy()

for i in range(n_display):
    # Original
    axes[0, i].imshow(X_test[i].numpy().reshape(8, 8), cmap='gray')
    axes[0, i].axis('off')
    if i == 0:
        axes[0, i].set_title('Original', fontsize=10)
    
    # Reconstructed
    axes[1, i].imshow(reconstructed[i].reshape(8, 8), cmap='gray')
    axes[1, i].axis('off')
    if i == 0:
        axes[1, i].set_title('Reconstructed', fontsize=10)

plt.tight_layout()
plt.show()
```

---

## Типи Autoencoders

### 1. Undercomplete Autoencoder

**Що це:** Latent dimension < input dimension (стандартний).

```python
class UndercompleteAE(nn.Module):
    def __init__(self):
        super(UndercompleteAE, self).__init__()
        
        # 784 → 128 → 64 → 32 (bottleneck)
        self.encoder = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # 32 → 64 → 128 → 784
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Sigmoid()  # Для pixel values [0,1]
        )
    
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon
```

**Використання:** Dimensionality reduction, compression.

### 2. Denoising Autoencoder (DAE)

**Що це:** Навчається відновлювати чистий вхід з зашумленого.

```python
class DenoisingAE(nn.Module):
    def __init__(self, input_dim=784, latent_dim=32):
        super(DenoisingAE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

# Навчання
def add_noise(x, noise_factor=0.3):
    noisy = x + noise_factor * torch.randn_like(x)
    return torch.clamp(noisy, 0., 1.)

model = DenoisingAE()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    # Додати шум до входу
    noisy_input = add_noise(X_train)
    
    # Forward (відновити чистий вхід з зашумленого)
    outputs = model(noisy_input)
    loss = criterion(outputs, X_train)  # Порівняти з чистим!
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**Застосування:**
- Видалення шуму з зображень
- Robust feature learning

### 3. Sparse Autoencoder

**Що це:** Додає sparsity constraint на latent representation.

```python
class SparseAE(nn.Module):
    def __init__(self, input_dim=784, latent_dim=64):
        super(SparseAE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

# Sparse loss
def sparse_loss(z, sparsity_target=0.05, sparsity_weight=1e-3):
    # L1 regularization на activation
    sparsity = torch.mean(torch.abs(z))
    return sparsity_weight * torch.abs(sparsity - sparsity_target)

# Навчання
for epoch in range(num_epochs):
    outputs, z = model(X_train)
    
    recon_loss = criterion(outputs, X_train)
    sparse_penalty = sparse_loss(z)
    
    total_loss = recon_loss + sparse_penalty
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
```

**Застосування:**
- Feature learning (мало активних нейронів)
- Interpretable features

### 4. Variational Autoencoder (VAE)

**Що це:** Generative model, який вчить probabilistic latent space.

```python
class VAE(nn.Module):
    def __init__(self, input_dim=784, latent_dim=20):
        super(VAE, self).__init__()
        
        # Encoder (outputs mean and log_var)
        self.fc1 = nn.Linear(input_dim, 400)
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, input_dim)
    
    def encode(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# VAE loss
def vae_loss(recon_x, x, mu, logvar):
    # Reconstruction loss
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + KLD

# Навчання
model = VAE()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    recon_batch, mu, logvar = model(X_train)
    loss = vae_loss(recon_batch, X_train, mu, logvar)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Генерація нових зразків
with torch.no_grad():
    z = torch.randn(64, 20)  # Sample from N(0,1)
    generated = model.decode(z).numpy()
    
    # Візуалізація згенерованих цифр
    fig, axes = plt.subplots(8, 8, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(generated[i].reshape(28, 28), cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    plt.show()
```

**Застосування:**
- Генерація нових зображень
- Interpolation в latent space
- Probabilistic modeling

### 5. Convolutional Autoencoder

**Що це:** Використовує CNN layers для зображень.

```python
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # 28x28 → 14x14
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # 14x14 → 7x7
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)  # 7x7 → 1x1 (bottleneck)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),  # 1x1 → 7x7
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # 7x7 → 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),   # 14x14 → 28x28
            nn.Sigmoid()
        )
    
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

# Використання
from torchvision import datasets, transforms

# MNIST dataset
transform = transforms.ToTensor()
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

model = ConvAutoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Навчання
for epoch in range(10):
    for data, _ in train_loader:
        outputs = model(data)
        loss = criterion(outputs, data)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
```

**Застосування:**
- Image compression
- Feature extraction для CNN
- Image-to-image tasks

---

## Застосування

### 1. Dimensionality Reduction

```python
# Autoencoder як альтернатива PCA

# 1. Навчити autoencoder
model = LinearAutoencoder(input_dim=784, latent_dim=50)
# ... train ...

# 2. Extract features
with torch.no_grad():
    X_reduced = model.encode(X_tensor).numpy()

print(f"Original: {X_tensor.shape}")   # (1797, 784)
print(f"Reduced: {X_reduced.shape}")    # (1797, 50)

# 3. Використати для downstream tasks
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()
clf.fit(X_reduced[:1000], y[:1000])
accuracy = clf.score(X_reduced[1000:], y[1000:])
print(f"Accuracy: {accuracy:.4f}")
```

### 2. Anomaly Detection

```python
# Навчити autoencoder на нормальних даних
# Аномалії матимуть високий reconstruction error

class AnomalyDetector:
    def __init__(self, model, threshold=None):
        self.model = model
        self.threshold = threshold
    
    def fit(self, X_normal):
        # Train autoencoder
        # ... training loop ...
        
        # Обчислити threshold на нормальних даних
        with torch.no_grad():
            recon = self.model(X_normal)
            errors = torch.mean((X_normal - recon) ** 2, dim=1)
            self.threshold = torch.quantile(errors, 0.95)  # 95th percentile
    
    def predict(self, X):
        with torch.no_grad():
            recon = self.model(X)
            errors = torch.mean((X - recon) ** 2, dim=1)
            
            # Аномалія якщо error > threshold
            is_anomaly = errors > self.threshold
            
        return is_anomaly.numpy(), errors.numpy()

# Використання
detector = AnomalyDetector(model)
detector.fit(X_normal_train)

is_anomaly, errors = detector.predict(X_test)

print(f"Anomalies detected: {is_anomaly.sum()}")

# Візуалізація
plt.hist(errors[~is_anomaly], bins=50, alpha=0.5, label='Normal')
plt.hist(errors[is_anomaly], bins=50, alpha=0.5, label='Anomaly')
plt.axvline(detector.threshold, color='red', linestyle='--', label='Threshold')
plt.xlabel('Reconstruction Error')
plt.ylabel('Count')
plt.legend()
plt.show()
```

### 3. Denoising

```python
# Видалити шум з зображень

# 1. Навчити denoising autoencoder
dae = DenoisingAE()
# ... train on (noisy_input, clean_target) ...

# 2. Denoise нових зображень
with torch.no_grad():
    noisy_images = add_noise(clean_images)
    denoised = dae(noisy_images)

# Візуалізація
fig, axes = plt.subplots(3, 10, figsize=(15, 5))
for i in range(10):
    axes[0, i].imshow(clean_images[i].reshape(28, 28), cmap='gray')
    axes[0, i].axis('off')
    if i == 0:
        axes[0, i].set_ylabel('Clean', fontsize=12)
    
    axes[1, i].imshow(noisy_images[i].reshape(28, 28), cmap='gray')
    axes[1, i].axis('off')
    if i == 0:
        axes[1, i].set_ylabel('Noisy', fontsize=12)
    
    axes[2, i].imshow(denoised[i].reshape(28, 28), cmap='gray')
    axes[2, i].axis('off')
    if i == 0:
        axes[2, i].set_ylabel('Denoised', fontsize=12)

plt.tight_layout()
plt.show()
```

### 4. Image Compression

```python
# Стиснення зображень

# Original: 28x28 = 784 pixels
# Latent: 32 dimensions

compression_ratio = 784 / 32  # 24.5x

# Encode
with torch.no_grad():
    latent = model.encode(images)  # (N, 32)

# Store only latent representation (compressed)
# Decode when needed
with torch.no_grad():
    reconstructed = model.decode(latent)  # (N, 784)

print(f"Compression ratio: {compression_ratio:.1f}x")
print(f"MSE: {torch.mean((images - reconstructed)**2).item():.6f}")
```

### 5. Feature Learning

```python
# Використати encoder як feature extractor

# 1. Навчити autoencoder (unsupervised)
model = ConvAutoencoder()
# ... train ...

# 2. Freeze encoder
for param in model.encoder.parameters():
    param.requires_grad = False

# 3. Додати classifier
class Classifier(nn.Module):
    def __init__(self, encoder):
        super(Classifier, self).__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(64, 10)  # 10 classes
    
    def forward(self, x):
        features = self.encoder(x)
        features = features.view(features.size(0), -1)  # Flatten
        logits = self.classifier(features)
        return logits

clf_model = Classifier(model.encoder)

# 4. Fine-tune classifier (supervised)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(clf_model.classifier.parameters(), lr=0.001)

# ... training loop ...
```

---

## Переваги та недоліки

### Переваги ✓

| Перевага | Пояснення |
|----------|-----------|
| **Нелінійність** | Вловлює складні нелінійні структури |
| **Flexibility** | Різні архітектури для різних задач |
| **Deep features** | Ієрархічні abstract representations |
| **Unsupervised** | Не потребує labels |
| **Versatility** | Reduction, denoising, generation, anomaly detection |
| **Scalability** | Працює на великих даних |

### Недоліки ✗

| Недолік | Пояснення |
|---------|-----------|
| **Складність** | Потребує hyperparameter tuning |
| **Обчислення** | Повільніше за PCA |
| **Дані** | Потребує багато зразків для навчання |
| **Не інтерпретовні** | Latent dimensions важко зрозуміти |
| **Локальні мінімуми** | Може застрягти при навчанні |
| **Overfitting** | Легко переобучитись на малих даних |

---

## Порівняння з іншими методами

| Метод | Лінійний | Supervised | Швидкість | Для ML | Генеративний |
|-------|----------|------------|-----------|--------|--------------|
| **PCA** | ✅ | ❌ | ⭐⭐⭐⭐⭐ | ✅ | ❌ |
| **Autoencoder** | ❌ | ❌ | ⭐⭐ | ✅ | ⚠️ (VAE) |
| **VAE** | ❌ | ❌ | ⭐⭐ | ⚠️ | ✅ |
| **t-SNE** | ❌ | ❌ | ⭐ | ❌ | ❌ |
| **UMAP** | ❌ | ❌ | ⭐⭐⭐ | ✅ | ❌ |

---

## Коли використовувати Autoencoders

### Ідеально підходить ✓

- **Нелінійні дані** — складні manifolds
- **Великі дані** — багато зразків для NN
- **Image data** — convolutional autoencoders
- **Denoising** — видалення шуму
- **Anomaly detection** — reconstruction error
- **Generation** — VAE для генерації
- **Deep features** — для downstream tasks

### Краще використати інше ✗

- **Малі дані** (< 1000) → PCA, t-SNE
- **Лінійні дані** → PCA простіший
- **Швидкість критична** → PCA
- **Інтерпретованість** → PCA
- **Тільки візуалізація** → t-SNE, UMAP швидше

---

## Практичні поради 💡

### 1. Почни з простої архітектури

```python
# ✅ Проста → складна
# Спочатку linear layers
model = LinearAutoencoder(input_dim=784, latent_dim=32)

# Якщо не вистачає → додай layers
# Якщо overfitting → зменш capacity
```

### 2. Використовуй правильну activation

```python
# Для pixel values [0, 1]
self.decoder = nn.Sequential(
    ...,
    nn.Sigmoid()  # ← output [0, 1]
)

# Для normalized data [-1, 1]
self.decoder = nn.Sequential(
    ...,
    nn.Tanh()  # ← output [-1, 1]
)

# Для будь-яких значень
self.decoder = nn.Sequential(
    ...,
    # No activation
)
```

### 3. Regularization для overfitting

```python
# Dropout
self.encoder = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Dropout(0.2),  # ← Dropout
    nn.Linear(128, 32)
)

# L2 regularization
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
```

### 4. Batch Normalization для stability

```python
self.encoder = nn.Sequential(
    nn.Linear(784, 128),
    nn.BatchNorm1d(128),  # ← BatchNorm
    nn.ReLU(),
    nn.Linear(128, 32)
)
```

### 5. Різні loss functions

```python
# MSE для regression-like
criterion = nn.MSELoss()

# Binary Cross-Entropy для binary inputs
criterion = nn.BCELoss()

# Комбінація
def combined_loss(recon, target, z):
    mse = nn.MSELoss()(recon, target)
    l1 = torch.mean(torch.abs(z))  # Sparsity
    return mse + 0.01 * l1
```

### 6. Learning rate scheduling

```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'min', patience=5
)

for epoch in range(num_epochs):
    # Train...
    loss = train_epoch()
    
    # Update learning rate
    scheduler.step(loss)
```

### 7. Early stopping

```python
best_loss = float('inf')
patience = 10
patience_counter = 0

for epoch in range(num_epochs):
    val_loss = validate()
    
    if val_loss < best_loss:
        best_loss = val_loss
        patience_counter = 0
        # Save model
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print("Early stopping!")
        break
```

### 8. Visualize latent space

```python
# Для 2D latent space
with torch.no_grad():
    z = model.encode(X_test)
    
plt.scatter(z[:, 0], z[:, 1], c=y_test, cmap='tab10')
plt.colorbar()
plt.title('2D Latent Space')
plt.show()
```

### 9. Interpolation

```python
# Інтерполяція між двома зразками
def interpolate(model, x1, x2, steps=10):
    with torch.no_grad():
        z1 = model.encode(x1.unsqueeze(0))
        z2 = model.encode(x2.unsqueeze(0))
        
        # Linear interpolation in latent space
        alphas = torch.linspace(0, 1, steps)
        interpolated = []
        
        for alpha in alphas:
            z = (1 - alpha) * z1 + alpha * z2
            x = model.decode(z)
            interpolated.append(x.squeeze())
    
    return torch.stack(interpolated)

# Візуалізація
interpolated = interpolate(model, img1, img2, steps=10)
fig, axes = plt.subplots(1, 10, figsize=(15, 2))
for i, ax in enumerate(axes):
    ax.imshow(interpolated[i].reshape(28, 28), cmap='gray')
    ax.axis('off')
plt.show()
```

### 10. Transfer learning

```python
# Використати pretrained encoder
pretrained = torch.load('pretrained_autoencoder.pth')
encoder = pretrained.encoder

# Freeze
for param in encoder.parameters():
    param.requires_grad = False

# Use for new task
```

---

## Поширені помилки ❌

### 1. Занадто складна архітектура для малих даних

```python
# ❌ Overfitting
model = nn.Sequential(
    nn.Linear(100, 1000),  # Занадто багато параметрів!
    nn.ReLU(),
    nn.Linear(1000, 500),
    nn.ReLU(),
    nn.Linear(500, 10)
)
# На 100 зразках → overfitting

# ✅ Проста архітектура
model = nn.Sequential(
    nn.Linear(100, 50),
    nn.ReLU(),
    nn.Linear(50, 10)
)
```

### 2. Неправильна activation function

```python
# ❌ Для pixel values [0, 1]
self.decoder = nn.Sequential(
    ...,
    # No activation → може бути негативні!
)

# ✅
self.decoder = nn.Sequential(
    ...,
    nn.Sigmoid()  # ← [0, 1]
)
```

### 3. Не використовувати validation set

```python
# ❌ Тільки train
for epoch in range(100):
    train_loss = train()

# ✅ Train + validation
for epoch in range(100):
    train_loss = train()
    val_loss = validate()
    
    if val_loss increasing:
        early_stop()
```

### 4. Забути normalize inputs

```python
# ❌ Без normalization
X_tensor = torch.FloatTensor(X)  # [0, 255]

# ✅ Normalize
X_normalized = X / 255.0  # [0, 1]
X_tensor = torch.FloatTensor(X_normalized)
```

### 5. Занадто високий learning rate

```python
# ❌ Divergence
optimizer = optim.Adam(model.parameters(), lr=0.1)

# ✅ Conservative
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

---

## Ресурси

- [PyTorch Autoencoder Tutorial](https://pytorch.org/tutorials/beginner/basics/autoencoders_tutorial.html)
- [VAE Paper (Kingma & Welling, 2013)](https://arxiv.org/abs/1312.6114)
- [Building Autoencoders in Keras](https://blog.keras.io/building-autoencoders-in-keras.html)
- [Stanford CS231n: Autoencoders](http://cs231n.stanford.edu/)

---

## Ключові висновки

> Autoencoders — це neural networks що навчаються стискати дані через bottleneck, створюючи compact representations для dimensionality reduction, denoising, anomaly detection та generation.

**Основні принципи:**
- **Encode-Decode:** Input → Compressed → Reconstructed
- **Unsupervised:** Навчається на unlabeled даних
- **Reconstruction:** Мінімізує ||x - x̂||²
- **Bottleneck:** Змушує мережу вчити compressed representation

**Архітектура:**
- **Encoder:** Стискає в latent space
- **Latent space:** Low-dimensional representation
- **Decoder:** Відновлює з latent

**Типи:**
- **Undercomplete:** Compression (latent < input)
- **Denoising:** Шум → чистий
- **Sparse:** L1 regularization
- **VAE:** Generative, probabilistic
- **Convolutional:** Для зображень

**Застосування:**
- Dimensionality reduction
- Anomaly detection (reconstruction error)
- Denoising
- Feature learning
- Generation (VAE)

**Переваги:**
- ✅ Нелінійні transformations
- ✅ Гнучкі архітектури
- ✅ Deep features

**Недоліки:**
- ❌ Потребує багато даних
- ❌ Складно налаштувати
- ❌ Повільніше за PCA

**Коли використовувати:**
- Нелінійні дані + багато зразків = Autoencoders ✓
- Лінійні дані / малі дані → PCA ✓
- Generation → VAE ✓
- Тільки візуалізація → t-SNE/UMAP ✓

**Найважливіше:**
- Почни просто, ускладнюй поступово
- Regularization проти overfitting
- Validation set обов'язковий
- Normalize inputs
- Visualize latent space

---

#ml #unsupervised-learning #dimensionality-reduction #autoencoders #neural-networks #deep-learning #vae #denoising #anomaly-detection
