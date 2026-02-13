# Information Theory (Теорія інформації)

## Що це?

**Information Theory** — це математична теорія, яка **вимірює кількість інформації** та **невизначеність** у даних. Використовується в ML, особливо в Decision Trees, для визначення **якості розділень** та **вибору ознак**.

**Головна ідея:** якщо подія дуже передбачувана (мало невизначеності) → мало інформації. Якщо подія несподівана (висока невизначеність) → багато інформації.

## Навіщо потрібно?

- 🌳 **Decision Trees** — вибір найкращого split
- 🎯 **Feature Selection** — які ознаки найінформативніші
- 📊 **Вимір чистоти** — наскільки однорідний набір даних
- 🔍 **Оцінка якості розділення** — чи добре розділили дані
- 💡 **Compression** — стиснення даних
- 🎲 **Ентропія** — міра невизначеності

## Коли використовувати?

**Потрібно:**
- Побудова Decision Trees (ID3, C4.5, CART)
- Feature importance в Random Forest
- Feature selection
- Вимір якості кластеризації

**Не потрібно:**
- Linear models (використовують інші критерії)
- Neural Networks (градієнтні методи)

---

## Ключові концепції

```
Information Theory для ML
│
├── Entropy (Ентропія)
│   └── Міра невизначеності/безладу
│
├── Information Gain (Приріст інформації)
│   └── Скільки інформації дає розділення
│
├── Gini Impurity (Індекс Джині)
│   └── Альтернатива ентропії
│
└── Gain Ratio
    └── Нормалізований Information Gain
```

---

# 1. Entropy (Ентропія)

## Що це?

**Entropy** — це **міра невизначеності** або **безладу** в наборі даних. Чим вище entropy, тим більше хаосу.

### Формула (Shannon Entropy)

$$H(S) = -\sum_{i=1}^{c} p_i \log_2(p_i)$$

де:
- $H(S)$ — ентропія множини $S$
- $c$ — кількість класів
- $p_i$ — ймовірність класу $i$
- $\log_2$ — логарифм по основі 2 (біти інформації)

### Інтуїція

```
Приклад: Монета

Чесна монета (50/50):
H = -(0.5 * log₂(0.5) + 0.5 * log₂(0.5))
  = -(0.5 * (-1) + 0.5 * (-1))
  = -(-0.5 - 0.5)
  = 1.0 біт

Максимальна невизначеність! Не знаємо, що випаде.

Нечесна монета (100% орел):
H = -(1.0 * log₂(1.0) + 0.0 * log₂(0.0))
  = -(0 + 0)  [використовуємо lim x→0 (x log x) = 0]
  = 0.0 біт

Жодної невизначеності! Завжди орел.
```

### Візуалізація Entropy

```python
import numpy as np
import matplotlib.pyplot as plt

def entropy(p):
    """Бінарна ентропія"""
    if p == 0 or p == 1:
        return 0
    return -(p * np.log2(p) + (1-p) * np.log2(1-p))

# Ймовірності
p_values = np.linspace(0.001, 0.999, 100)
entropy_values = [entropy(p) for p in p_values]

# Візуалізація
plt.figure(figsize=(10, 6))
plt.plot(p_values, entropy_values, linewidth=3, color='blue')
plt.xlabel('p (Ймовірність класу 1)', fontsize=12)
plt.ylabel('Entropy H(p)', fontsize=12)
plt.title('Binary Entropy Function', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Max entropy = 1.0')
plt.axvline(x=0.5, color='green', linestyle='--', alpha=0.5, label='p = 0.5')
plt.legend(fontsize=11)

# Анотації
plt.annotate('Максимум\nH = 1.0\n(p = 0.5)', 
            xy=(0.5, 1.0), xytext=(0.3, 0.85),
            fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
            arrowprops=dict(arrowstyle='->', color='green', lw=2))

plt.annotate('Чиста класифікація\nH = 0\n(p = 0 або p = 1)', 
            xy=(0.1, 0.05), xytext=(0.25, 0.3),
            fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7),
            arrowprops=dict(arrowstyle='->', color='red', lw=2))

plt.tight_layout()
plt.show()
```

**Інтерпретація графіка:**
- Максимум при $p = 0.5$ (50/50) — найбільша невизначеність
- Мінімум при $p = 0$ або $p = 1$ — повна визначеність
- Симетрична функція

### Приклад обчислення

```python
import numpy as np

def calculate_entropy(labels):
    """
    Обчислити ентропію для списку міток класів
    
    Parameters:
    -----------
    labels : array-like
        Мітки класів
        
    Returns:
    --------
    float : ентропія (в бітах)
    """
    # Унікальні класи та їх частоти
    unique, counts = np.unique(labels, return_counts=True)
    
    # Ймовірності
    probabilities = counts / len(labels)
    
    # Ентропія
    entropy = 0
    for p in probabilities:
        if p > 0:  # Уникаємо log(0)
            entropy -= p * np.log2(p)
    
    return entropy

# Приклад 1: Ідеально збалансовані класи (50/50)
labels_balanced = ['A'] * 50 + ['B'] * 50
H_balanced = calculate_entropy(labels_balanced)
print(f"Приклад 1 (50/50):")
print(f"  Labels: 50 A, 50 B")
print(f"  Entropy: {H_balanced:.4f} біт")

# Приклад 2: Несбалансовані класи (80/20)
labels_imbalanced = ['A'] * 80 + ['B'] * 20
H_imbalanced = calculate_entropy(labels_imbalanced)
print(f"\nПриклад 2 (80/20):")
print(f"  Labels: 80 A, 20 B")
print(f"  Entropy: {H_imbalanced:.4f} біт")

# Приклад 3: Чистий набір (100/0)
labels_pure = ['A'] * 100
H_pure = calculate_entropy(labels_pure)
print(f"\nПриклад 3 (100/0):")
print(f"  Labels: 100 A, 0 B")
print(f"  Entropy: {H_pure:.4f} біт")

# Приклад 4: Три класи (40/30/30)
labels_multiclass = ['A'] * 40 + ['B'] * 30 + ['C'] * 30
H_multiclass = calculate_entropy(labels_multiclass)
print(f"\nПриклад 4 (3 класи: 40/30/30):")
print(f"  Labels: 40 A, 30 B, 30 C")
print(f"  Entropy: {H_multiclass:.4f} біт")
```

**Вивід:**
```
Приклад 1 (50/50):
  Labels: 50 A, 50 B
  Entropy: 1.0000 біт  ← Максимум для 2 класів

Приклад 2 (80/20):
  Labels: 80 A, 20 B
  Entropy: 0.7219 біт  ← Менше невизначеності

Приклад 3 (100/0):
  Labels: 100 A, 0 B
  Entropy: 0.0000 біт  ← Жодної невизначеності

Приклад 4 (3 класи: 40/30/30):
  Labels: 40 A, 30 B, 30 C
  Entropy: 1.5710 біт  ← Більше для 3 класів
```

### Максимальна ентропія

Для $c$ класів, максимальна ентропія:

$$H_{\max} = \log_2(c)$$

```
2 класи:   H_max = log₂(2) = 1.0 біт
3 класи:   H_max = log₂(3) ≈ 1.585 біт
4 класи:   H_max = log₂(4) = 2.0 біт
10 класів: H_max = log₂(10) ≈ 3.322 біт
```

### Візуалізація розподілів

```python
import matplotlib.pyplot as plt
import numpy as np

# Різні розподіли
distributions = [
    {'name': 'Pure (100/0)', 'values': [100, 0], 'entropy': 0.0},
    {'name': 'Very skewed (90/10)', 'values': [90, 10], 'entropy': 0.469},
    {'name': 'Skewed (80/20)', 'values': [80, 20], 'entropy': 0.722},
    {'name': 'Imbalanced (70/30)', 'values': [70, 30], 'entropy': 0.881},
    {'name': 'Balanced (50/50)', 'values': [50, 50], 'entropy': 1.0}
]

fig, axes = plt.subplots(1, 5, figsize=(18, 4))

for idx, dist in enumerate(distributions):
    axes[idx].bar(['Class A', 'Class B'], dist['values'], 
                  color=['blue', 'orange'], alpha=0.7)
    axes[idx].set_title(f"{dist['name']}\nH = {dist['entropy']:.3f}", 
                       fontsize=11, fontweight='bold')
    axes[idx].set_ylim([0, 105])
    axes[idx].set_ylabel('Count' if idx == 0 else '')
    axes[idx].grid(True, alpha=0.3, axis='y')

plt.suptitle('Entropy for Different Class Distributions', 
            fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
```

---

## 2. Information Gain (Приріст інформації)

### Що це?

**Information Gain** — це **зменшення ентропії** після розділення даних за певною ознакою. Показує, **скільки інформації** ми отримали від розділення.

### Формула

$$\text{IG}(S, A) = H(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} H(S_v)$$

де:
- $H(S)$ — ентропія до розділення
- $A$ — ознака, за якою розділяємо
- $S_v$ — підмножина даних, де $A = v$
- $\frac{|S_v|}{|S|}$ — частка зразків у підмножині

**Інтуїція:**
```
IG = Entropy(до) - Weighted_Average(Entropy(після))

IG > 0 → Розділення зменшило невизначеність ✓
IG = 0 → Розділення не дало нічого
IG < 0 → Неможливо (ентропія не може зрости)
```

### Приклад: Play Tennis

```
Dataset: Грати в теніс чи ні?

Outlook | Temperature | Humidity | Windy | Play?
--------|-------------|----------|-------|------
Sunny   | Hot         | High     | False | No
Sunny   | Hot         | High     | True  | No
Overcast| Hot         | High     | False | Yes
Rain    | Mild        | High     | False | Yes
Rain    | Cool        | Normal   | False | Yes
Rain    | Cool        | Normal   | True  | No
Overcast| Cool        | Normal   | True  | Yes
Sunny   | Mild        | High     | False | No
Sunny   | Cool        | Normal   | False | Yes
Rain    | Mild        | Normal   | False | Yes
Sunny   | Mild        | Normal   | True  | Yes
Overcast| Mild        | High     | True  | Yes
Overcast| Hot         | Normal   | False | Yes
Rain    | Mild        | High     | True  | No

Всього: 14 зразків
Yes: 9 (грати)
No: 5 (не грати)
```

### Крок 1: Ентропія до розділення

```python
import numpy as np

# Дані
total = 14
yes = 9
no = 5

# Ймовірності
p_yes = yes / total
p_no = no / total

# Ентропія
H_before = -(p_yes * np.log2(p_yes) + p_no * np.log2(p_no))

print(f"До розділення:")
print(f"  Yes: {yes}, No: {no}")
print(f"  p(Yes): {p_yes:.3f}, p(No): {p_no:.3f}")
print(f"  H(S) = {H_before:.4f} біт")

# Вивід:
# До розділення:
#   Yes: 9, No: 5
#   p(Yes): 0.643, p(No): 0.357
#   H(S) = 0.9403 біт
```

### Крок 2: Розділення за Outlook

```python
# Розділення за Outlook
outlook_splits = {
    'Sunny': {'yes': 2, 'no': 3},      # 5 зразків
    'Overcast': {'yes': 4, 'no': 0},   # 4 зразки
    'Rain': {'yes': 3, 'no': 2}        # 5 зразків
}

def calculate_split_entropy(splits, total):
    """Обчислити зважену ентропію після розділення"""
    weighted_entropy = 0
    
    for value, counts in splits.items():
        n = counts['yes'] + counts['no']
        weight = n / total
        
        if n == 0:
            continue
        
        p_yes = counts['yes'] / n
        p_no = counts['no'] / n
        
        # Ентропія цієї підмножини
        H = 0
        if p_yes > 0:
            H -= p_yes * np.log2(p_yes)
        if p_no > 0:
            H -= p_no * np.log2(p_no)
        
        weighted_entropy += weight * H
        
        print(f"  {value:10s}: {counts['yes']} Yes, {counts['no']} No → "
              f"H = {H:.4f}, weight = {weight:.3f}")
    
    return weighted_entropy

print(f"\nРозділення за Outlook:")
H_after_outlook = calculate_split_entropy(outlook_splits, total)
print(f"  Weighted H(S|Outlook) = {H_after_outlook:.4f}")

# Information Gain
IG_outlook = H_before - H_after_outlook
print(f"  IG(S, Outlook) = {H_before:.4f} - {H_after_outlook:.4f} = {IG_outlook:.4f}")

# Вивід:
# Розділення за Outlook:
#   Sunny     : 2 Yes, 3 No → H = 0.9710, weight = 0.357
#   Overcast  : 4 Yes, 0 No → H = 0.0000, weight = 0.286
#   Rain      : 3 Yes, 2 No → H = 0.9710, weight = 0.357
#   Weighted H(S|Outlook) = 0.6935
#   IG(S, Outlook) = 0.9403 - 0.6935 = 0.2467
```

### Крок 3: Порівняння з іншими ознаками

```python
# Розділення за Temperature
temp_splits = {
    'Hot': {'yes': 2, 'no': 2},
    'Mild': {'yes': 4, 'no': 2},
    'Cool': {'yes': 3, 'no': 1}
}

print(f"\nРозділення за Temperature:")
H_after_temp = calculate_split_entropy(temp_splits, total)
IG_temp = H_before - H_after_temp
print(f"  IG(S, Temperature) = {IG_temp:.4f}")

# Розділення за Humidity
humidity_splits = {
    'High': {'yes': 3, 'no': 4},
    'Normal': {'yes': 6, 'no': 1}
}

print(f"\nРозділення за Humidity:")
H_after_humidity = calculate_split_entropy(humidity_splits, total)
IG_humidity = H_before - H_after_humidity
print(f"  IG(S, Humidity) = {IG_humidity:.4f}")

# Розділення за Windy
windy_splits = {
    'False': {'yes': 6, 'no': 2},
    'True': {'yes': 3, 'no': 3}
}

print(f"\nРозділення за Windy:")
H_after_windy = calculate_split_entropy(windy_splits, total)
IG_windy = H_before - H_after_windy
print(f"  IG(S, Windy) = {IG_windy:.4f}")

# Порівняння
print(f"\n{'='*50}")
print("ПОРІВНЯННЯ INFORMATION GAIN:")
print(f"{'='*50}")
gains = {
    'Outlook': IG_outlook,
    'Humidity': IG_humidity,
    'Windy': IG_windy,
    'Temperature': IG_temp
}

for feature, gain in sorted(gains.items(), key=lambda x: x[1], reverse=True):
    print(f"  {feature:15s}: {gain:.4f}")

print(f"\n✅ Найкраща ознака для розділення: Outlook (IG = {IG_outlook:.4f})")
```

### Візуалізація Information Gain

```python
import matplotlib.pyplot as plt

features = list(gains.keys())
ig_values = list(gains.values())

plt.figure(figsize=(10, 6))
bars = plt.bar(features, ig_values, color=['green', 'orange', 'blue', 'purple'], alpha=0.7)

# Виділити найкращу
max_idx = np.argmax(ig_values)
bars[max_idx].set_color('red')
bars[max_idx].set_alpha(1.0)

plt.ylabel('Information Gain', fontsize=12)
plt.title('Information Gain for Each Feature', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# Анотації
for i, (feature, value) in enumerate(zip(features, ig_values)):
    plt.text(i, value + 0.01, f'{value:.4f}', 
            ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
plt.tight_layout()
plt.show()
```

### Decision Tree: Використання IG

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
import pandas as pd

# Створити DataFrame
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 
                'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 
                'Overcast', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool',
                   'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal',
                'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Windy': ['False', 'True', 'False', 'False', 'False', 'True', 'True',
             'False', 'False', 'False', 'True', 'True', 'False', 'True'],
    'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes',
            'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)

# Encode categorical variables
from sklearn.preprocessing import LabelEncoder

le_dict = {}
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

X = df.drop('Play', axis=1)
y = df['Play']

# Decision Tree з criterion='entropy' (Information Gain)
dt_entropy = DecisionTreeClassifier(criterion='entropy', random_state=42)
dt_entropy.fit(X, y)

# Візуалізація дерева
plt.figure(figsize=(20, 10))
plot_tree(dt_entropy, 
         feature_names=['Outlook', 'Temperature', 'Humidity', 'Windy'],
         class_names=['No', 'Yes'],
         filled=True,
         rounded=True,
         fontsize=10)
plt.title('Decision Tree (criterion=entropy, Information Gain)', 
         fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Feature importances
importances = dt_entropy.feature_importances_
feature_names = ['Outlook', 'Temperature', 'Humidity', 'Windy']

print("\nFeature Importances (based on Information Gain):")
for name, importance in zip(feature_names, importances):
    print(f"  {name:15s}: {importance:.4f}")
```

---

## 3. Gini Impurity (Індекс Джині)

### Що це?

**Gini Impurity** — це альтернативна метрика **чистоти** набору даних. Вимірює ймовірність **неправильної класифікації** випадкового зразка.

### Формула

$$\text{Gini}(S) = 1 - \sum_{i=1}^{c} p_i^2$$

де:
- $p_i$ — ймовірність класу $i$
- $c$ — кількість класів

### Інтуїція

```
Gini Impurity = Ймовірність помилкової класифікації

Приклад: Монета (50/50)

Gini = 1 - (0.5² + 0.5²)
     = 1 - (0.25 + 0.25)
     = 1 - 0.5
     = 0.5

Якщо випадково взяти зразок і випадково присвоїти мітку,
ймовірність помилки = 0.5 (50%)

Чистий набір (100/0):
Gini = 1 - (1.0² + 0.0²)
     = 1 - 1.0
     = 0.0

Ймовірність помилки = 0 (жодних помилок)
```

### Порівняння Entropy vs Gini

```python
import numpy as np
import matplotlib.pyplot as plt

def gini_impurity(p):
    """Бінарна Gini impurity"""
    return 1 - (p**2 + (1-p)**2)

def entropy(p):
    """Бінарна ентропія"""
    if p == 0 or p == 1:
        return 0
    return -(p * np.log2(p) + (1-p) * np.log2(1-p))

# Ймовірності
p_values = np.linspace(0.001, 0.999, 100)
gini_values = [gini_impurity(p) for p in p_values]
entropy_values = [entropy(p) for p in p_values]

# Normalize entropy to [0, 1] для порівняння
entropy_normalized = [e / max(entropy_values) for e in entropy_values]

# Візуалізація
plt.figure(figsize=(12, 6))

plt.plot(p_values, gini_values, linewidth=3, label='Gini Impurity', color='blue')
plt.plot(p_values, entropy_normalized, linewidth=3, label='Entropy (normalized)', 
        color='red', linestyle='--')

plt.xlabel('p (Ймовірність класу 1)', fontsize=12)
plt.ylabel('Impurity', fontsize=12)
plt.title('Gini Impurity vs Entropy', fontsize=14, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

# Анотації
plt.axvline(x=0.5, color='green', linestyle=':', alpha=0.5, linewidth=2)
plt.text(0.5, 0.05, 'p = 0.5\n(максимум)', ha='center', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

plt.tight_layout()
plt.show()

# Порівняння значень
print("Порівняння Entropy vs Gini:")
print("="*50)
for p in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
    e = entropy(p)
    g = gini_impurity(p)
    print(f"p = {p:.2f}: Entropy = {e:.4f}, Gini = {g:.4f}")
```

### Обчислення Gini

```python
def calculate_gini(labels):
    """
    Обчислити Gini impurity для списку міток
    
    Parameters:
    -----------
    labels : array-like
        Мітки класів
        
    Returns:
    --------
    float : Gini impurity
    """
    # Унікальні класи та їх частоти
    unique, counts = np.unique(labels, return_counts=True)
    
    # Ймовірності
    probabilities = counts / len(labels)
    
    # Gini impurity
    gini = 1 - np.sum(probabilities ** 2)
    
    return gini

# Приклади
labels_50_50 = ['A'] * 50 + ['B'] * 50
labels_80_20 = ['A'] * 80 + ['B'] * 20
labels_pure = ['A'] * 100

print("Gini Impurity:")
print(f"  50/50:  {calculate_gini(labels_50_50):.4f}")
print(f"  80/20:  {calculate_gini(labels_80_20):.4f}")
print(f"  100/0:  {calculate_gini(labels_pure):.4f}")

# Вивід:
# Gini Impurity:
#   50/50:  0.5000  ← Максимум для 2 класів
#   80/20:  0.3200
#   100/0:  0.0000  ← Чистий набір
```

### Gini Gain (аналог Information Gain)

```python
def calculate_gini_gain(labels_before, splits):
    """
    Обчислити Gini Gain після розділення
    
    Parameters:
    -----------
    labels_before : array-like
        Мітки до розділення
    splits : dict
        Словник {значення: список_міток}
        
    Returns:
    --------
    float : Gini gain
    """
    # Gini до розділення
    gini_before = calculate_gini(labels_before)
    
    # Зважена Gini після розділення
    n_total = len(labels_before)
    weighted_gini_after = 0
    
    for value, labels_subset in splits.items():
        weight = len(labels_subset) / n_total
        gini_subset = calculate_gini(labels_subset)
        weighted_gini_after += weight * gini_subset
    
    # Gini Gain
    gini_gain = gini_before - weighted_gini_after
    
    return gini_gain, gini_before, weighted_gini_after

# Приклад: Tennis dataset, розділення за Outlook
labels_all = ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 
             'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']

outlook_splits = {
    'Sunny': ['No', 'No', 'No', 'Yes', 'Yes'],
    'Overcast': ['Yes', 'Yes', 'Yes', 'Yes'],
    'Rain': ['Yes', 'Yes', 'No', 'Yes', 'No']
}

gini_gain, gini_before, gini_after = calculate_gini_gain(labels_all, outlook_splits)

print(f"Gini Gain для Outlook:")
print(f"  Gini до:    {gini_before:.4f}")
print(f"  Gini після: {gini_after:.4f}")
print(f"  Gini Gain:  {gini_gain:.4f}")
```

### Decision Tree з Gini

```python
from sklearn.tree import DecisionTreeClassifier

# Той самий Tennis dataset
# dt_gini використовує Gini impurity (default)
dt_gini = DecisionTreeClassifier(criterion='gini', random_state=42)
dt_gini.fit(X, y)

# Порівняння Entropy vs Gini
dt_entropy = DecisionTreeClassifier(criterion='entropy', random_state=42)
dt_entropy.fit(X, y)

print("Feature Importances:")
print("="*50)
print(f"{'Feature':<15} {'Gini':>10} {'Entropy':>10}")
print("="*50)

for i, name in enumerate(['Outlook', 'Temperature', 'Humidity', 'Windy']):
    print(f"{name:<15} {dt_gini.feature_importances_[i]:>10.4f} "
          f"{dt_entropy.feature_importances_[i]:>10.4f}")
```

---

## 4. Gain Ratio (Нормалізований Information Gain)

### Проблема Information Gain

**Проблема:** Information Gain **схильний до ознак з багатьма значеннями**.

```python
# Приклад: Ознака "ID" (унікальна для кожного зразка)

# Dataset: 14 зразків, 9 Yes, 5 No
# ID: 1, 2, 3, ..., 14 (унікальні)

# Розділення за ID:
# Кожен split має 1 зразок → Entropy = 0 для кожного

# IG(S, ID) = H(S) - 0 = H(S) = максимальний!
# Але ID не має предиктивної сили! ❌
```

### Рішення: Gain Ratio

$$\text{GainRatio}(S, A) = \frac{\text{IG}(S, A)}{\text{SplitInfo}(S, A)}$$

де:

$$\text{SplitInfo}(S, A) = -\sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} \log_2 \frac{|S_v|}{|S|}$$

**SplitInfo** — це ентропія розподілу зразків по значеннях ознаки (без врахування класів).

### Інтуїція

```
SplitInfo карає ознаки з багатьма значеннями:

Ознака з 2 значеннями (50/50):
  SplitInfo = -(0.5 log₂(0.5) + 0.5 log₂(0.5)) = 1.0

Ознака з 14 значеннями (по 1 зразку):
  SplitInfo = -14 * (1/14 log₂(1/14)) ≈ 3.807

Gain Ratio = IG / SplitInfo
Для ID: GainRatio буде значно менше!
```

### Обчислення

```python
def calculate_gain_ratio(labels_all, splits):
    """
    Обчислити Gain Ratio
    
    Parameters:
    -----------
    labels_all : array-like
        Всі мітки
    splits : dict
        Розділення {значення: список_міток}
        
    Returns:
    --------
    float : Gain Ratio
    """
    # Information Gain
    H_before = calculate_entropy(labels_all)
    
    n_total = len(labels_all)
    weighted_H_after = 0
    
    # SplitInfo
    split_info = 0
    
    for value, labels_subset in splits.items():
        n_subset = len(labels_subset)
        weight = n_subset / n_total
        
        # Для IG
        H_subset = calculate_entropy(labels_subset)
        weighted_H_after += weight * H_subset
        
        # Для SplitInfo
        if weight > 0:
            split_info -= weight * np.log2(weight)
    
    IG = H_before - weighted_H_after
    
    # Gain Ratio
    if split_info == 0:
        return 0
    
    gain_ratio = IG / split_info
    
    return gain_ratio, IG, split_info

# Приклад 1: Outlook (3 значення)
outlook_splits = {
    'Sunny': ['No', 'No', 'No', 'Yes', 'Yes'],
    'Overcast': ['Yes', 'Yes', 'Yes', 'Yes'],
    'Rain': ['Yes', 'Yes', 'No', 'Yes', 'No']
}

labels_all = ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 
             'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']

gr_outlook, ig_outlook, si_outlook = calculate_gain_ratio(labels_all, outlook_splits)

print("Outlook:")
print(f"  IG:         {ig_outlook:.4f}")
print(f"  SplitInfo:  {si_outlook:.4f}")
print(f"  GainRatio:  {gr_outlook:.4f}")

# Приклад 2: ID (14 унікальних значень)
id_splits = {f'ID_{i}': [labels_all[i]] for i in range(len(labels_all))}

gr_id, ig_id, si_id = calculate_gain_ratio(labels_all, id_splits)

print("\nID (14 unique values):")
print(f"  IG:         {ig_id:.4f}")  # Максимальний!
print(f"  SplitInfo:  {si_id:.4f}")  # Дуже великий
print(f"  GainRatio:  {gr_id:.4f}")  # Нормалізований

# Вивід:
# Outlook:
#   IG:         0.2467
#   SplitInfo:  1.5774
#   GainRatio:  0.1564

# ID:
#   IG:         0.9403  ← Максимальний!
#   SplitInfo:  3.8074  ← Дуже великий
#   GainRatio:  0.2469  ← Менше за Outlook!
```

---

## Порівняння метрик

### Таблиця порівняння

| Метрика | Формула | Range | Коли використовувати |
|---------|---------|-------|---------------------|
| **Entropy** | $-\sum p_i \log_2 p_i$ | [0, $\log_2(c)$] | ID3, C4.5 algorithms |
| **Gini** | $1 - \sum p_i^2$ | [0, $1-\frac{1}{c}$] | CART algorithm (sklearn default) |
| **Information Gain** | $H(S) - H(S\|A)$ | [0, $H(S)$] | Feature selection |
| **Gain Ratio** | $\frac{IG}{SplitInfo}$ | [0, 1] | C4.5 (уникає bias до багатьох значень) |

### Візуальне порівняння

```python
import matplotlib.pyplot as plt
import numpy as np

# Генерація даних
p_values = np.linspace(0.001, 0.999, 100)

entropy_values = [entropy(p) for p in p_values]
gini_values = [gini_impurity(p) for p in p_values]

# Normalize для порівняння
entropy_norm = np.array(entropy_values) / max(entropy_values)
gini_norm = np.array(gini_values) / max(gini_values)

# Різниця
diff = entropy_norm - gini_norm

# Візуалізація
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Subplot 1: Обидві криві
axes[0].plot(p_values, entropy_norm, linewidth=2, label='Entropy (normalized)', color='blue')
axes[0].plot(p_values, gini_norm, linewidth=2, label='Gini (normalized)', color='red')
axes[0].set_xlabel('p', fontsize=12)
axes[0].set_ylabel('Impurity (normalized)', fontsize=12)
axes[0].set_title('Entropy vs Gini Impurity', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Subplot 2: Різниця
axes[1].plot(p_values, diff, linewidth=2, color='green')
axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1)
axes[1].fill_between(p_values, 0, diff, where=(diff > 0), alpha=0.3, color='green', label='Entropy > Gini')
axes[1].fill_between(p_values, 0, diff, where=(diff < 0), alpha=0.3, color='red', label='Gini > Entropy')
axes[1].set_xlabel('p', fontsize=12)
axes[1].set_ylabel('Difference (Entropy - Gini)', fontsize=12)
axes[1].set_title('Difference between Metrics', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Спостереження
print("Спостереження:")
print("1. Обидві метрики мають максимум при p=0.5")
print("2. Обидві мінімальні (0) при p=0 або p=1")
print("3. Entropy трохи більша за Gini в середньому діапазоні")
print("4. На практиці обидві дають схожі результати")
```

### Швидкість обчислення

```python
import time

# Великий dataset
np.random.seed(42)
n = 100000
labels = np.random.choice(['A', 'B', 'C'], size=n)

# Benchmark Entropy
start = time.time()
for _ in range(100):
    H = calculate_entropy(labels)
time_entropy = time.time() - start

# Benchmark Gini
start = time.time()
for _ in range(100):
    G = calculate_gini(labels)
time_gini = time.time() - start

print("Час обчислення (100 ітерацій):")
print(f"  Entropy: {time_entropy:.4f} сек")
print(f"  Gini:    {time_gini:.4f} сек")
print(f"\nGini швидша на {(time_entropy/time_gini - 1)*100:.1f}%")

# Зазвичай Gini трохи швидша (без логарифмів)
```

---

## Практичне застосування

### 1. Побудова Decision Tree вручну

```python
import numpy as np
import pandas as pd

class SimpleDecisionTree:
    """Спрощене дерево рішень з Information Gain"""
    
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = None
    
    def entropy(self, labels):
        """Обчислити ентропію"""
        unique, counts = np.unique(labels, return_counts=True)
        probabilities = counts / len(labels)
        return -np.sum([p * np.log2(p) for p in probabilities if p > 0])
    
    def information_gain(self, X, y, feature):
        """Обчислити Information Gain для ознаки"""
        # Ентропія до розділення
        H_before = self.entropy(y)
        
        # Унікальні значення ознаки
        values = np.unique(X[:, feature])
        
        # Зважена ентропія після розділення
        weighted_H_after = 0
        for value in values:
            mask = X[:, feature] == value
            subset_y = y[mask]
            weight = len(subset_y) / len(y)
            weighted_H_after += weight * self.entropy(subset_y)
        
        # Information Gain
        return H_before - weighted_H_after
    
    def find_best_feature(self, X, y):
        """Знайти найкращу ознаку для розділення"""
        n_features = X.shape[1]
        
        gains = []
        for feature in range(n_features):
            gain = self.information_gain(X, y, feature)
            gains.append(gain)
        
        return np.argmax(gains), max(gains)
    
    def build_tree(self, X, y, depth=0):
        """Рекурсивно побудувати дерево"""
        # Base cases
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            # Leaf node: повернути найчастіший клас
            unique, counts = np.unique(y, return_counts=True)
            return unique[np.argmax(counts)]
        
        # Знайти найкращу ознаку
        best_feature, best_gain = self.find_best_feature(X, y)
        
        if best_gain == 0:
            # Немає покращення
            unique, counts = np.unique(y, return_counts=True)
            return unique[np.argmax(counts)]
        
        # Створити вузол
        tree = {
            'feature': best_feature,
            'gain': best_gain,
            'children': {}
        }
        
        # Розділити та рекурсивно побудувати піддерева
        values = np.unique(X[:, best_feature])
        for value in values:
            mask = X[:, best_feature] == value
            subset_X = X[mask]
            subset_y = y[mask]
            
            tree['children'][value] = self.build_tree(subset_X, subset_y, depth + 1)
        
        return tree
    
    def fit(self, X, y):
        """Навчити дерево"""
        self.tree = self.build_tree(X, y)
        return self
    
    def predict_single(self, x, tree):
        """Передбачення для одного зразка"""
        if not isinstance(tree, dict):
            # Leaf node
            return tree
        
        feature_value = x[tree['feature']]
        
        if feature_value in tree['children']:
            return self.predict_single(x, tree['children'][feature_value])
        else:
            # Значення не бачили під час навчання
            # Повернути найчастіший клас серед children
            leaves = []
            for child in tree['children'].values():
                if not isinstance(child, dict):
                    leaves.append(child)
            return max(set(leaves), key=leaves.count) if leaves else 0
    
    def predict(self, X):
        """Передбачення для багатьох зразків"""
        return np.array([self.predict_single(x, self.tree) for x in X])

# Використання
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Дані
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)

# Дискретизація (для спрощення)
# Розбити на 3 bins
for i in range(X_train.shape[1]):
    X_train[:, i] = np.digitize(X_train[:, i], 
                                bins=np.percentile(X_train[:, i], [33, 66]))
    X_test[:, i] = np.digitize(X_test[:, i], 
                               bins=np.percentile(X_train[:, i], [33, 66]))

# Навчити дерево
tree = SimpleDecisionTree(max_depth=3)
tree.fit(X_train, y_train)

# Передбачення
y_pred = tree.predict(X_test)

# Оцінка
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Структура дерева
print("\nTree structure:")
print(tree.tree)
```

### 2. Feature Selection з Information Gain

```python
from sklearn.datasets import load_breast_cancer
import pandas as pd

# Дані
cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = cancer.target

# Обчислити IG для всіх ознак
def calculate_feature_ig(X, y, n_bins=5):
    """Обчислити Information Gain для всіх ознак"""
    igs = {}
    
    for feature in X.columns:
        # Дискретизація continuous ознаки
        X_binned = pd.cut(X[feature], bins=n_bins, labels=False)
        
        # Information Gain
        H_before = calculate_entropy(y)
        
        weighted_H_after = 0
        for bin_value in range(n_bins):
            mask = X_binned == bin_value
            if mask.sum() == 0:
                continue
            
            subset_y = y[mask]
            weight = len(subset_y) / len(y)
            weighted_H_after += weight * calculate_entropy(subset_y)
        
        ig = H_before - weighted_H_after
        igs[feature] = ig
    
    return igs

# Обчислити IG
igs = calculate_feature_ig(X, y)

# Відсортувати
igs_sorted = sorted(igs.items(), key=lambda x: x[1], reverse=True)

# Топ-10
print("Top 10 Features by Information Gain:")
print("="*60)
for i, (feature, ig) in enumerate(igs_sorted[:10], 1):
    print(f"{i:2d}. {feature:30s}: {ig:.6f}")

# Візуалізація
import matplotlib.pyplot as plt

features = [f for f, _ in igs_sorted[:15]]
values = [ig for _, ig in igs_sorted[:15]]

plt.figure(figsize=(12, 6))
plt.barh(features, values, color='steelblue', alpha=0.7)
plt.xlabel('Information Gain', fontsize=12)
plt.title('Top 15 Features by Information Gain', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()
```

---

## Поширені помилки ❌

### 1. Забувати про log(0)

```python
# ❌ ПОГАНО
def entropy_bad(labels):
    unique, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    return -np.sum(probabilities * np.log2(probabilities))  # log(0) = -inf!

# ✅ ДОБРЕ
def entropy_good(labels):
    unique, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    return -np.sum([p * np.log2(p) for p in probabilities if p > 0])
```

### 2. Плутати Entropy з Gini

```python
# Entropy: -Σ p log(p)
# Gini:    1 - Σ p²

# НЕ плутати!
```

### 3. Не нормалізувати Gain Ratio

```python
# Information Gain схильний до багатьох значень
# Використовуй Gain Ratio для ознак з великою кількістю унікальних значень
```

---

## Практичні поради 💡

### 1. Entropy vs Gini: що вибрати?

```python
# На практиці обидві дають схожі результати

# Entropy:
# ✅ Теоретично більш обґрунтована
# ❌ Повільніша (логарифми)

# Gini:
# ✅ Швидша (без логарифмів)
# ✅ Default у sklearn
# ✅ Проста інтерпретація

# Рекомендація: використовуй Gini (sklearn default)
```

### 2. Information Gain для Feature Selection

```python
from sklearn.feature_selection import mutual_info_classif

# mutual_info_classif = Information Gain для continuous features
mi_scores = mutual_info_classif(X, y, random_state=42)

# Топ ознаки
top_features = np.argsort(mi_scores)[::-1][:10]
print("Top 10 features by Mutual Information:")
for idx in top_features:
    print(f"  {X.columns[idx]}: {mi_scores[idx]:.4f}")
```

### 3. Візуалізуй розділення

```python
import matplotlib.pyplot as plt

def visualize_split(X, y, feature, threshold):
    """Візуалізувати якість розділення"""
    
    # Розділити
    left_mask = X[:, feature] <= threshold
    right_mask = ~left_mask
    
    # Обчислити метрики
    H_before = calculate_entropy(y)
    H_left = calculate_entropy(y[left_mask])
    H_right = calculate_entropy(y[right_mask])
    
    weight_left = left_mask.sum() / len(y)
    weight_right = right_mask.sum() / len(y)
    
    H_after = weight_left * H_left + weight_right * H_right
    IG = H_before - H_after
    
    # Візуалізація
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Subplot 1: Розподіл до
    axes[0].hist(y, bins=np.unique(y), alpha=0.7, color='gray')
    axes[0].set_title(f'Before Split\nEntropy = {H_before:.4f}', fontweight='bold')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Count')
    
    # Subplot 2: Ліва гілка
    axes[1].hist(y[left_mask], bins=np.unique(y), alpha=0.7, color='blue')
    axes[1].set_title(f'Left ({left_mask.sum()} samples)\nEntropy = {H_left:.4f}', 
                     fontweight='bold')
    axes[1].set_xlabel('Class')
    
    # Subplot 3: Права гілка
    axes[2].hist(y[right_mask], bins=np.unique(y), alpha=0.7, color='red')
    axes[2].set_title(f'Right ({right_mask.sum()} samples)\nEntropy = {H_right:.4f}', 
                     fontweight='bold')
    axes[2].set_xlabel('Class')
    
    plt.suptitle(f'Information Gain = {IG:.4f}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# Приклад
from sklearn.datasets import load_iris
iris = load_iris()
visualize_split(iris.data, iris.target, feature=2, threshold=2.5)
```

---

## Реальний приклад: Comprehensive Analysis

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

print("="*70)
print("INFORMATION THEORY: COMPREHENSIVE ANALYSIS")
print("="*70)

# Дані
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target

print(f"\nDataset: {len(y)} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
print(f"Class distribution: {np.bincount(y)}")

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ============================================
# 1. ENTROPY ANALYSIS
# ============================================
print("\n" + "="*70)
print("1. ENTROPY ANALYSIS")
print("="*70)

H_train = calculate_entropy(y_train)
H_test = calculate_entropy(y_test)

print(f"\nEntropy:")
print(f"  Train set: {H_train:.4f} bits")
print(f"  Test set:  {H_test:.4f} bits")
print(f"  Max entropy (3 classes): {np.log2(3):.4f} bits")

# ============================================
# 2. INFORMATION GAIN FOR EACH FEATURE
# ============================================
print("\n" + "="*70)
print("2. INFORMATION GAIN PER FEATURE")
print("="*70)

# Обчислити IG для кожної ознаки (з дискретизацією)
igs = {}
ginis = {}

for feature in X.columns:
    # Дискретизація
    X_binned = pd.cut(X_train[feature], bins=5, labels=False, duplicates='drop')
    
    # Information Gain
    weighted_H = 0
    for bin_val in range(5):
        mask = X_binned == bin_val
        if mask.sum() == 0:
            continue
        subset_y = y_train[mask]
        weight = len(subset_y) / len(y_train)
        weighted_H += weight * calculate_entropy(subset_y)
    
    ig = H_train - weighted_H
    igs[feature] = ig

# Топ-10
igs_sorted = sorted(igs.items(), key=lambda x: x[1], reverse=True)

print("\nTop 10 Features by Information Gain:")
for i, (feature, ig) in enumerate(igs_sorted[:10], 1):
    print(f"  {i:2d}. {feature:30s}: {ig:.6f}")

# ============================================
# 3. DECISION TREES: ENTROPY VS GINI
# ============================================
print("\n" + "="*70)
print("3. DECISION TREES: ENTROPY VS GINI")
print("="*70)

# Tree з Entropy
dt_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
dt_entropy.fit(X_train, y_train)
acc_entropy_train = dt_entropy.score(X_train, y_train)
acc_entropy_test = dt_entropy.score(X_test, y_test)

# Tree з Gini
dt_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
dt_gini.fit(X_train, y_train)
acc_gini_train = dt_gini.score(X_train, y_train)
acc_gini_test = dt_gini.score(X_test, y_test)

print("\nDecision Tree Performance:")
print(f"  Entropy - Train: {acc_entropy_train:.4f}, Test: {acc_entropy_test:.4f}")
print(f"  Gini    - Train: {acc_gini_train:.4f}, Test: {acc_gini_test:.4f}")

# Feature importances
print("\nFeature Importances (Entropy):")
for feature, importance in sorted(zip(X.columns, dt_entropy.feature_importances_), 
                                  key=lambda x: x[1], reverse=True)[:5]:
    print(f"  {feature:30s}: {importance:.4f}")

# ============================================
# 4. VISUALIZATIONS
# ============================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Subplot 1: Information Gain bar plot
features_top10 = [f for f, _ in igs_sorted[:10]]
ig_top10 = [ig for _, ig in igs_sorted[:10]]

axes[0, 0].barh(features_top10, ig_top10, color='steelblue', alpha=0.7)
axes[0, 0].set_xlabel('Information Gain', fontsize=11)
axes[0, 0].set_title('Top 10 Features by Information Gain', 
                    fontsize=13, fontweight='bold')
axes[0, 0].invert_yaxis()
axes[0, 0].grid(True, alpha=0.3, axis='x')

# Subplot 2: Feature Importances comparison
features_imp = X.columns
imp_entropy = dt_entropy.feature_importances_
imp_gini = dt_gini.feature_importances_

# Топ-10
top_features_idx = np.argsort(imp_entropy)[-10:][::-1]

x_pos = np.arange(len(top_features_idx))
width = 0.35

axes[0, 1].barh(x_pos - width/2, imp_entropy[top_features_idx], 
               width, label='Entropy', alpha=0.8)
axes[0, 1].barh(x_pos + width/2, imp_gini[top_features_idx], 
               width, label='Gini', alpha=0.8)
axes[0, 1].set_yticks(x_pos)
axes[0, 1].set_yticklabels([features_imp[i] for i in top_features_idx])
axes[0, 1].set_xlabel('Feature Importance', fontsize=11)
axes[0, 1].set_title('Feature Importances: Entropy vs Gini', 
                    fontsize=13, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].invert_yaxis()
axes[0, 1].grid(True, alpha=0.3, axis='x')

# Subplot 3: Entropy vs Gini curve
p_vals = np.linspace(0.001, 0.999, 100)
entropy_vals = [entropy(p) for p in p_vals]
gini_vals = [gini_impurity(p) for p in p_vals]

axes[1, 0].plot(p_vals, entropy_vals, linewidth=2, label='Entropy', color='blue')
axes[1, 0].plot(p_vals, [g * np.log2(np.e) for g in gini_vals], linewidth=2, 
               label='Gini (scaled)', color='red', linestyle='--')
axes[1, 0].set_xlabel('p (Probability of class 1)', fontsize=11)
axes[1, 0].set_ylabel('Impurity', fontsize=11)
axes[1, 0].set_title('Entropy vs Gini Impurity', fontsize=13, fontweight='bold')
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(True, alpha=0.3)

# Subplot 4: Decision Tree (Entropy)
from sklearn.tree import plot_tree

plot_tree(dt_entropy, 
         feature_names=X.columns,
         class_names=wine.target_names,
         filled=True,
         rounded=True,
         fontsize=8,
         ax=axes[1, 1])
axes[1, 1].set_title('Decision Tree (Entropy, max_depth=3)', 
                    fontsize=13, fontweight='bold')

plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"✅ Dataset entropy: {H_train:.4f} bits")
print(f"✅ Most informative feature: {igs_sorted[0][0]} (IG={igs_sorted[0][1]:.4f})")
print(f"✅ Entropy-based tree accuracy: {acc_entropy_test:.4f}")
print(f"✅ Gini-based tree accuracy: {acc_gini_test:.4f}")
print("="*70)
```

---

## Підсумкова таблиця

| Концепція | Формула | Діапазон | Використання |
|-----------|---------|----------|--------------|
| **Entropy** | $-\sum p_i \log_2 p_i$ | [0, $\log_2(c)$] | Міра невизначеності |
| **Gini** | $1 - \sum p_i^2$ | [0, $1-\frac{1}{c}$] | Альтернатива ентропії |
| **IG** | $H(S) - H(S\|A)$ | [0, $H(S)$] | Вибір ознаки для split |
| **Gain Ratio** | $\frac{IG}{SplitInfo}$ | [0, 1] | Нормалізований IG |

---

## Пов'язані теми

- [[Decision_Trees]] — основне застосування Information Theory
- [[Random_Forest]] — Feature importance через IG
- [[Feature_Selection]] — вибір ознак за IG
- [[CART]] — використання Gini impurity
- [[ID3_C45]] — алгоритми на основі Entropy

## Ресурси

- [Shannon Entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory))
- [Information Gain](https://en.wikipedia.org/wiki/Information_gain_in_decision_trees)
- [Elements of Information Theory](https://www.wiley.com/en-us/Elements+of+Information+Theory%2C+2nd+Edition-p-9780471241959)

---

## Ключові висновки

> Information Theory надає математичний фреймворк для вимірювання невизначеності та інформації в даних. Entropy, Information Gain, та Gini Impurity — ключові метрики для побудови Decision Trees.

**Ключові концепції:**
- **Entropy** — міра невизначеності (більше хаосу = більше ентропії)
- **Information Gain** — скільки інформації дає розділення
- **Gini Impurity** — альтернатива ентропії (швидша, схожі результати)
- **Gain Ratio** — нормалізований IG (уникає bias до багатьох значень)

**Практичне застосування:**
- **Decision Trees** — вибір найкращого split
- **Feature Selection** — найінформативніші ознаки
- **Feature Importance** — ранжування ознак

**Entropy vs Gini:**
- Обидві дають схожі результати
- Gini швидша (без логарифмів)
- Entropy теоретично більш обґрунтована
- **Рекомендація:** використовуй Gini (sklearn default)

**Формули для запам'ятовування:**
```
Entropy:  H = -Σ pᵢ log₂(pᵢ)
Gini:     G = 1 - Σ pᵢ²
IG:       IG = H(before) - H(after)
```

---

#ml #information-theory #entropy #information-gain #gini-impurity #decision-trees #feature-selection
