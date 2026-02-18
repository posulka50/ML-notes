# Naive Bayes (Наївний Байєс)

## Що це?

**Naive Bayes** — це сімейство простих **probabilistic алгоритмів** supervised learning, які базуються на **теоремі Байєса** з "наївним" припущенням про **незалежність ознак**.

**Головна ідея:** обчислити ймовірність належності до кожного класу на основі ознак, припускаючи, що всі ознаки незалежні між собою.

## Навіщо потрібен?

- ⚡ **Швидкість** — дуже швидке навчання та передбачення
- 📊 **Простота** — легко зрозуміти та реалізувати
- 🎯 **Baseline** — чудовий початок для класифікації
- 📧 **Text classification** — стандарт для spam detection, sentiment analysis
- 💡 **Малі дані** — працює навіть з невеликими датасетами
- 🔧 **Probabilistic** — дає калібровані ймовірності
- 🚀 **Real-time** — швидкі online predictions

## Коли використовувати?

**Потрібно:**
- **Text classification** — spam detection, sentiment analysis, document categorization
- **Швидкий baseline** — перша модель для спроби
- **Малі датасети** — працює навіть з малою кількістю даних
- **Real-time predictions** — дуже швидкі передбачення
- **Probabilistic outputs** — потрібні ймовірності класів
- **Multi-class classification** — природно працює з багатьма класами
- **Високорозмірні дані** — не страждає від curse of dimensionality

**Не потрібно:**
- Ознаки **сильно корелюють** між собою → порушення незалежності
- Потрібна **максимальна точність** → Tree-based, SVM, Neural Networks
- **Складні взаємодії** між ознаками → Tree-based, Neural Networks
- Числові ознаки з нестандартним розподілом → інші алгоритми

---

## Теорема Байєса

### Формула

$$P(C|X) = \frac{P(X|C) \cdot P(C)}{P(X)}$$

де:
- $P(C|X)$ — **posterior probability** (ймовірність класу $C$ за умови ознак $X$)
- $P(X|C)$ — **likelihood** (ймовірність ознак $X$ за умови класу $C$)
- $P(C)$ — **prior probability** (апріорна ймовірність класу $C$)
- $P(X)$ — **evidence** (ймовірність ознак $X$)

### Інтуїція

**Приклад: Діагностика хвороби**

- $P(\text{Хворий}|\text{Позитивний тест})$ — яка ймовірність, що людина хвора, якщо тест позитивний?

**За теоремою Байєса:**

$$P(\text{Хворий}|\text{+}) = \frac{P(\text{+}|\text{Хворий}) \cdot P(\text{Хворий})}{P(\text{+})}$$

де:
- $P(\text{+}|\text{Хворий})$ — чутливість тесту (якщо хворий, яка ймовірність позитивного результату?)
- $P(\text{Хворий})$ — скільки людей хворі в популяції?
- $P(\text{+})$ — скільки тестів позитивні загалом?

---

## "Naive" припущення

### Припущення про незалежність

**Naive Bayes припускає:** всі ознаки **незалежні** між собою за умови класу.

$$P(X|C) = P(x_1, x_2, ..., x_n | C) = \prod_{i=1}^{n} P(x_i | C)$$

**Чому "naive" (наївне)?**

В реальності ознаки майже завжди **залежні**, але ми припускаємо незалежність для спрощення.

### Приклад

**Email spam detection:**

Ознаки: слова "free", "win", "prize"

**Naive припущення:**

$$P(\text{"free", "win", "prize"}|\text{Spam}) = P(\text{"free"}|\text{Spam}) \cdot P(\text{"win"}|\text{Spam}) \cdot P(\text{"prize"}|\text{Spam})$$

**Реальність:** Ці слова часто йдуть разом у spam → вони **залежні**!

**Чому все одно працює?**

Навіть з порушенням припущення, Naive Bayes часто дає правильний **порядок** ймовірностей → правильна класифікація! ✓

---

## Алгоритм Naive Bayes

### Training Phase

1. **Обчислити prior probabilities для кожного класу:**

$$P(C_k) = \frac{\text{count}(C_k)}{n}$$

2. **Обчислити likelihood для кожної ознаки за умови класу:**

$$P(x_i | C_k)$$

(спосіб обчислення залежить від типу даних — див. нижче)

### Prediction Phase

1. **Для нового зразка $X = [x_1, x_2, ..., x_n]$ обчислити posterior для кожного класу:**

$$P(C_k | X) \propto P(C_k) \cdot \prod_{i=1}^{n} P(x_i | C_k)$$

(можна опустити $P(X)$, бо він однаковий для всіх класів)

2. **Обрати клас з найвищою posterior probability:**

$$\hat{y} = \arg\max_{C_k} P(C_k | X)$$

### Log-trick для стабільності

**Проблема:** Множення багатьох малих ймовірностей → **numerical underflow**.

**Рішення:** Використовувати **логарифми**:

$$\log P(C_k | X) = \log P(C_k) + \sum_{i=1}^{n} \log P(x_i | C_k)$$

Сума логарифмів замість добутку ймовірностей!

---

## Типи Naive Bayes

## 1. Gaussian Naive Bayes

### Для continuous (числових) ознак

**Припущення:** Ознаки розподілені **нормально** (Gaussian) за умови класу.

$$P(x_i | C_k) = \frac{1}{\sqrt{2\pi\sigma_{k,i}^2}} \exp\left(-\frac{(x_i - \mu_{k,i})^2}{2\sigma_{k,i}^2}\right)$$

де:
- $\mu_{k,i}$ — середнє ознаки $i$ для класу $k$
- $\sigma_{k,i}^2$ — variance ознаки $i$ для класу $k$

### Training

**Для кожного класу $k$ та ознаки $i$:**

$$\mu_{k,i} = \frac{1}{n_k} \sum_{x \in C_k} x_i$$

$$\sigma_{k,i}^2 = \frac{1}{n_k} \sum_{x \in C_k} (x_i - \mu_{k,i})^2$$

### Візуалізація

```
Feature distribution by class:

    P(x|C)
      |        Class A         Class B
      |          ╱╲              ╱╲
      |         ╱  ╲            ╱  ╲
      |        ╱    ╲          ╱    ╲
      |       ╱      ╲        ╱      ╲
      |______╱________╲______╱________╲_____ x
           μ_A      μ_B
```

### Код

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Дані
iris = load_iris()
X = iris.data
y = iris.target

# Розділення
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Gaussian Naive Bayes
gnb = GaussianNB()

# Навчання (дуже швидко!)
gnb.fit(X_train, y_train)

# Передбачення
y_pred = gnb.predict(X_test)
y_pred_proba = gnb.predict_proba(X_test)

# Оцінка
print("=== Gaussian Naive Bayes ===")
print(f"Train Accuracy: {gnb.score(X_train, y_train):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Параметри моделі
print("\n=== Model Parameters ===")
print(f"Class priors: {gnb.class_prior_}")
print(f"\nMeans (μ) for class 0:")
print(gnb.theta_[0])
print(f"\nVariances (σ²) for class 0:")
print(gnb.var_[0])
```

### Коли використовувати Gaussian NB

✅ Continuous numerical features
✅ Ознаки приблизно нормально розподілені
✅ Малі/середні датасети
✅ Швидкий baseline

---

## 2. Multinomial Naive Bayes

### Для discrete counts (лічильників)

**Припущення:** Ознаки представляють **частоти** або **counts** (наприклад, word counts в документі).

$$P(x_i | C_k) = \frac{\text{count}(x_i, C_k) + \alpha}{\sum_{j} \text{count}(x_j, C_k) + \alpha \cdot n_{features}}$$

де $\alpha$ — **smoothing parameter** (Laplace smoothing).

### Laplace Smoothing

**Проблема:** Якщо слово ніколи не зустрічалось у класі → $P(x_i|C_k) = 0$ → posterior = 0!

**Рішення:** Додаємо $\alpha$ (зазвичай 1) до всіх counts:

$$P(x_i | C_k) = \frac{\text{count}(x_i, C_k) + \alpha}{N_k + \alpha \cdot V}$$

де:
- $N_k$ — загальна кількість слів у класі $k$
- $V$ — розмір vocabulary

**Ефект:**
- $\alpha = 0$ → no smoothing (може бути 0 ймовірності)
- $\alpha = 1$ — **Laplace smoothing** (стандарт)
- $\alpha > 1$ — більше згладжування

### Приклад: Text Classification

**Документи:**
```
Doc 1 (Sport): "football match goal score"
Doc 2 (Sport): "basketball game score"
Doc 3 (Tech):  "python code software"
```

**Vocabulary:** [football, match, goal, score, basketball, game, python, code, software]

**Word counts для Sport:**
- football: 1
- match: 1
- goal: 1
- score: 2
- basketball: 1
- game: 1
- python: 0 ← НЕ ЗУСТРІЧАЄТЬСЯ!

**З Laplace smoothing (α=1):**

$$P(\text{"python"}|\text{Sport}) = \frac{0 + 1}{7 + 1 \cdot 9} = \frac{1}{16} \neq 0$$ ✓

### Код

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Дані (text)
texts = [
    "free money win prize",
    "meeting schedule tomorrow",
    "limited offer click now",
    "project update report",
    "win lottery free cash",
    "team meeting agenda"
]
labels = [1, 0, 1, 0, 1, 0]  # 1=spam, 0=ham

# Vectorization (Bag of Words)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

print("=== Bag of Words ===")
print(f"Vocabulary: {vectorizer.get_feature_names_out()}")
print(f"Shape: {X.shape}")
print(f"Example document vector:\n{X[0].toarray()}")

# Розділення
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.3, random_state=42
)

# Multinomial Naive Bayes
mnb = MultinomialNB(alpha=1.0)  # Laplace smoothing

# Навчання
mnb.fit(X_train, y_train)

# Передбачення
y_pred = mnb.predict(X_test)
y_pred_proba = mnb.predict_proba(X_test)

# Оцінка
print("\n=== Multinomial Naive Bayes ===")
print(f"Train Accuracy: {mnb.score(X_train, y_train):.4f}")
print(f"Test Accuracy: {mnb.score(X_test, y_test):.4f}")

# Параметри
print(f"\nClass priors: {mnb.class_prior_}")
print(f"Feature log probabilities shape: {mnb.feature_log_prob_.shape}")

# Передбачення для нового документу
new_text = ["free prize money"]
X_new = vectorizer.transform(new_text)
pred = mnb.predict(X_new)
pred_proba = mnb.predict_proba(X_new)

print(f"\nNew document: '{new_text[0]}'")
print(f"Prediction: {'Spam' if pred[0] == 1 else 'Ham'}")
print(f"Probabilities: Ham={pred_proba[0][0]:.4f}, Spam={pred_proba[0][1]:.4f}")
```

### Коли використовувати Multinomial NB

✅ **Text classification** (найпопулярніше!)
✅ **Document categorization**
✅ **Spam detection**
✅ **Sentiment analysis**
✅ Дані у вигляді counts/frequencies
✅ Sparse високорозмірні дані

---

## 3. Bernoulli Naive Bayes

### Для binary features (0/1)

**Припущення:** Ознаки **бінарні** (присутність/відсутність).

$$P(x_i | C_k) = P(i | C_k) \cdot x_i + (1 - P(i | C_k)) \cdot (1 - x_i)$$

де $P(i | C_k)$ — ймовірність того, що ознака $i$ присутня в класі $k$.

**Різниця з Multinomial:**
- **Multinomial:** враховує **скільки разів** слово зустрічається
- **Bernoulli:** враховує тільки **чи зустрічається** (є/немає)

### Приклад

**Документ:** "free free money prize"

**Multinomial representation:**
- free: 2
- money: 1
- prize: 1

**Bernoulli representation:**
- free: 1 (присутнє)
- money: 1 (присутнє)
- prize: 1 (присутнє)

### Код

```python
from sklearn.naive_bayes import BernoulliNB

# Бінаризація даних (0/1)
from sklearn.preprocessing import Binarizer

binarizer = Binarizer()
X_binary = binarizer.fit_transform(X.toarray())

# Bernoulli Naive Bayes
bnb = BernoulliNB(alpha=1.0)
bnb.fit(X_binary, labels)

print(f"Bernoulli NB Accuracy: {bnb.score(X_binary, labels):.4f}")
```

### Коли використовувати Bernoulli NB

✅ Binary features (presence/absence)
✅ Text classification з binary representation
✅ Коли важливіше **чи присутнє** слово, а не **скільки разів**
✅ Короткі документи

---

## Порівняння типів Naive Bayes

| Тип | Дані | Розподіл | Використання | Приклад |
|-----|------|----------|--------------|---------|
| **Gaussian** | Continuous | Нормальний | Числові ознаки | Iris classification |
| **Multinomial** | Counts | Multinomial | Text (word counts) | Spam detection |
| **Bernoulli** | Binary | Bernoulli | Text (presence) | Short text classification |

### Вибір типу

```
                Тип даних?
                /         \
        Continuous       Discrete
            |               |
       Gaussian NB      Binary або Counts?
                        /              \
                   Binary              Counts
                      |                  |
                Bernoulli NB      Multinomial NB
```

---

## TF-IDF з Naive Bayes

### TF-IDF (Term Frequency - Inverse Document Frequency)

**Краще за простий Bag of Words** для текстів:

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \cdot \text{IDF}(t)$$

де:
- $\text{TF}(t, d)$ — частота терміну $t$ в документі $d$
- $\text{IDF}(t) = \log\frac{N}{n_t}$ — inverse document frequency

**Ідея:** Рідкісні слова важливіші за частотні ("the", "a", "is").

### Код

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF замість Bag of Words
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_tfidf = tfidf_vectorizer.fit_transform(texts)

# Multinomial NB з TF-IDF
mnb_tfidf = MultinomialNB()
mnb_tfidf.fit(X_tfidf, labels)

print(f"MNB with TF-IDF Accuracy: {mnb_tfidf.score(X_tfidf, labels):.4f}")
```

**Примітка:** TF-IDF може давати негативні значення після центрування → використовуй **MultinomialNB**, але будь обережний з параметрами.

---

## Простий приклад: Email Spam Detection

### Дані

```
Email 1 (Spam):  "Free money! Win prize now! Click here!"
Email 2 (Ham):   "Meeting tomorrow at 10am. See you there."
Email 3 (Spam):  "Limited offer! Free cash prize!"
Email 4 (Ham):   "Project report attached. Please review."

New email: "Free prize available now!"
```

### Крок 1: Обчислити Prior Probabilities

$$P(\text{Spam}) = \frac{2}{4} = 0.5$$
$$P(\text{Ham}) = \frac{2}{4} = 0.5$$

### Крок 2: Обчислити Likelihoods

**Vocabulary:** {free, money, win, prize, click, meeting, project, ...}

**Word counts у Spam:**
- free: 2, money: 1, prize: 2, ...
- Total words in Spam: 14

**Word counts у Ham:**
- free: 0, money: 0, prize: 0, meeting: 1, project: 1, ...
- Total words in Ham: 12

**З Laplace smoothing (α=1):**

$$P(\text{"free"}|\text{Spam}) = \frac{2 + 1}{14 + V} \approx 0.15$$
$$P(\text{"free"}|\text{Ham}) = \frac{0 + 1}{12 + V} \approx 0.05$$

(де $V$ — розмір vocabulary)

### Крок 3: Класифікувати новий email

**New:** "Free prize available now!"

$$P(\text{Spam}|\text{New}) \propto P(\text{Spam}) \cdot P(\text{"free"}|\text{Spam}) \cdot P(\text{"prize"}|\text{Spam}) \cdot ...$$

$$P(\text{Ham}|\text{New}) \propto P(\text{Ham}) \cdot P(\text{"free"}|\text{Ham}) \cdot P(\text{"prize"}|\text{Ham}) \cdot ...$$

**Результат:** $P(\text{Spam}|\text{New}) > P(\text{Ham}|\text{New})$ → **Spam!** ✓

---

## Складний приклад: Multi-class Document Classification

```python
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Синтетичні дані (document classification)
np.random.seed(42)

# Категорії та характерні слова
categories = {
    'Sport': ['football', 'basketball', 'game', 'match', 'score', 'team', 'player', 'win'],
    'Tech': ['python', 'code', 'software', 'computer', 'algorithm', 'data', 'programming'],
    'Politics': ['election', 'president', 'vote', 'government', 'law', 'congress', 'policy']
}

def generate_document(category):
    words = np.random.choice(categories[category], size=np.random.randint(5, 15))
    return ' '.join(words)

# Генерація датасету
n_per_class = 200
documents = []
labels = []

for cat_idx, (category, _) in enumerate(categories.items()):
    for _ in range(n_per_class):
        documents.append(generate_document(category))
        labels.append(cat_idx)

print("="*70)
print("MULTINOMIAL NAIVE BAYES FOR DOCUMENT CLASSIFICATION")
print("="*70)
print(f"Dataset: {len(documents)} documents")
print(f"Categories: {list(categories.keys())}")
print(f"Documents per category: {n_per_class}")

# Розділення
X_train, X_test, y_train, y_test = train_test_split(
    documents, labels, test_size=0.2, random_state=42, stratify=labels
)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(
    max_features=500,
    ngram_range=(1, 2),  # Unigrams + Bigrams
    min_df=2
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"\nTF-IDF shape: {X_train_tfidf.shape}")
print(f"Vocabulary size: {len(vectorizer.get_feature_names_out())}")

# 1. Базова модель
print("\n" + "="*70)
print("1. BASELINE MODEL")
print("="*70)

mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_train_tfidf, y_train)

y_pred = mnb.predict(X_test_tfidf)

print(f"Train Accuracy: {mnb.score(X_train_tfidf, y_train):.4f}")
print(f"Test Accuracy: {mnb.score(X_test_tfidf, y_test):.4f}")

# 2. Cross-validation
print("\n" + "="*70)
print("2. CROSS-VALIDATION")
print("="*70)

cv_scores = cross_val_score(mnb, X_train_tfidf, y_train, cv=5)
print(f"CV Scores: {cv_scores}")
print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# 3. Підбір alpha (smoothing)
print("\n" + "="*70)
print("3. TUNING ALPHA (SMOOTHING)")
print("="*70)

alphas = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
train_scores = []
test_scores = []

for alpha in alphas:
    mnb_alpha = MultinomialNB(alpha=alpha)
    mnb_alpha.fit(X_train_tfidf, y_train)
    train_scores.append(mnb_alpha.score(X_train_tfidf, y_train))
    test_scores.append(mnb_alpha.score(X_test_tfidf, y_test))

optimal_alpha = alphas[np.argmax(test_scores)]
print(f"Optimal alpha: {optimal_alpha}")
print(f"Best test score: {max(test_scores):.4f}")

# 4. Фінальна модель
print("\n" + "="*70)
print("4. FINAL MODEL EVALUATION")
print("="*70)

mnb_final = MultinomialNB(alpha=optimal_alpha)
mnb_final.fit(X_train_tfidf, y_train)

y_pred_final = mnb_final.predict(X_test_tfidf)
y_pred_proba = mnb_final.predict_proba(X_test_tfidf)

print(f"Test Accuracy: {mnb_final.score(X_test_tfidf, y_test):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_final,
                          target_names=list(categories.keys())))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred_final)
print(cm)

# 5. Top words per category
print("\n" + "="*70)
print("5. TOP WORDS PER CATEGORY")
print("="*70)

feature_names = vectorizer.get_feature_names_out()

for cat_idx, category in enumerate(categories.keys()):
    # Отримати log probabilities
    log_prob = mnb_final.feature_log_prob_[cat_idx]
    
    # Топ 10 слів
    top_10_idx = np.argsort(log_prob)[-10:][::-1]
    
    print(f"\n{category}:")
    for idx in top_10_idx:
        print(f"  {feature_names[idx]}: {np.exp(log_prob[idx]):.4f}")

# 6. Передбачення для нових документів
print("\n" + "="*70)
print("6. PREDICTIONS FOR NEW DOCUMENTS")
print("="*70)

new_docs = [
    "football match score game",
    "python programming code algorithm",
    "election president vote government"
]

X_new = vectorizer.transform(new_docs)
predictions = mnb_final.predict(X_new)
probabilities = mnb_final.predict_proba(X_new)

for doc, pred, proba in zip(new_docs, predictions, probabilities):
    pred_category = list(categories.keys())[pred]
    print(f"\nDocument: '{doc}'")
    print(f"Prediction: {pred_category}")
    print("Probabilities:")
    for cat_idx, category in enumerate(categories.keys()):
        print(f"  {category}: {proba[cat_idx]:.4f}")

# Візуалізації
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Alpha vs Accuracy
axes[0, 0].plot(alphas, train_scores, 'o-', linewidth=2, label='Train')
axes[0, 0].plot(alphas, test_scores, 's-', linewidth=2, label='Test')
axes[0, 0].axvline(x=optimal_alpha, color='red', linestyle='--',
                   label=f'Optimal α={optimal_alpha}')
axes[0, 0].set_xlabel('Alpha (Smoothing)', fontsize=12)
axes[0, 0].set_ylabel('Accuracy', fontsize=12)
axes[0, 0].set_title('Accuracy vs Alpha', fontsize=14, fontweight='bold')
axes[0, 0].set_xscale('log')
axes[0, 0].legend(fontsize=11)
axes[0, 0].grid(True, alpha=0.3)

# 2. Confusion Matrix
import seaborn as sns
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1],
            xticklabels=categories.keys(),
            yticklabels=categories.keys())
axes[0, 1].set_xlabel('Predicted', fontsize=12)
axes[0, 1].set_ylabel('Actual', fontsize=12)
axes[0, 1].set_title('Confusion Matrix', fontsize=14, fontweight='bold')

# 3. Class probabilities distribution
for cat_idx, category in enumerate(categories.keys()):
    probs = y_pred_proba[:, cat_idx]
    axes[1, 0].hist(probs, bins=20, alpha=0.5, label=category)
axes[1, 0].set_xlabel('Predicted Probability', fontsize=12)
axes[1, 0].set_ylabel('Frequency', fontsize=12)
axes[1, 0].set_title('Distribution of Predicted Probabilities',
                    fontsize=14, fontweight='bold')
axes[1, 0].legend(fontsize=11)
axes[1, 0].grid(True, alpha=0.3)

# 4. Cross-validation scores
axes[1, 1].bar(range(len(cv_scores)), cv_scores)
axes[1, 1].axhline(y=cv_scores.mean(), color='red', linestyle='--',
                   label=f'Mean: {cv_scores.mean():.4f}')
axes[1, 1].set_xlabel('Fold', fontsize=12)
axes[1, 1].set_ylabel('Accuracy', fontsize=12)
axes[1, 1].set_title('Cross-Validation Scores', fontsize=14, fontweight='bold')
axes[1, 1].legend(fontsize=11)
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
```

---

## Переваги та недоліки

### Переваги ✓

| Перевага | Пояснення |
|----------|-----------|
| **Швидкість** | Дуже швидке навчання та передбачення |
| **Простота** | Легко зрозуміти та реалізувати |
| **Малі дані** | Працює навіть з малими датасетами |
| **Probabilistic** | Дає калібровані ймовірності |
| **Multi-class** | Природно працює з багатьма класами |
| **Online learning** | Легко оновлювати на нових даних |
| **High-dimensional** | Не страждає від curse of dimensionality |
| **Baseline** | Чудова стартова модель |
| **Text classification** | Стандарт для spam, sentiment analysis |
| **Обробка missing** | Gaussian NB природно обробляє |

### Недоліки ✗

| Недолік | Пояснення |
|---------|-----------|
| **Naive припущення** | Ознаки майже завжди залежні |
| **Точність** | Поступається складнішим моделям |
| **Нормальність** | Gaussian NB вимагає нормального розподілу |
| **Не optimal** | Decision boundary не оптимізована |
| **Numerical features** | Gaussian NB може погано з multimodal |
| **Категоріальні ознаки** | Потребує encoding |
| **Feature engineering** | Обмежені можливості |

---

## Порівняння з іншими алгоритмами

### Naive Bayes vs Logistic Regression

| Критерій | Naive Bayes | Logistic Regression |
|----------|-------------|---------------------|
| **Assumptions** | Незалежність ознак | Лінійна розділимість |
| **Training** | O(n·d) | O(n·d·k) iterations |
| **Prediction** | O(d) | O(d) |
| **Швидкість** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Точність** | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Малі дані** | ✅ Добре | ⚠️ Може overfitting |
| **Text** | ✅ Відмінно | ✅ Добре |

### Naive Bayes vs SVM

| Критерій | Naive Bayes | SVM |
|----------|-------------|-----|
| **Training** | O(n·d) | O(n²) to O(n³) |
| **Prediction** | O(d) | O(n_sv·d) |
| **Швидкість** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Точність** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Text** | ✅ Стандарт | ✅ Працює |
| **Простота** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

### Naive Bayes vs Random Forest

| Критерій | Naive Bayes | Random Forest |
|----------|-------------|---------------|
| **Training** | O(n·d) | O(n·log(n)·d·T) |
| **Prediction** | O(d) | O(T·log(n)) |
| **Швидкість** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Точність** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Interpretability** | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Feature scaling** | Не критично | Не потрібна |

---

## Коли використовувати Naive Bayes

### Ідеально підходить ✓

- **Text classification** — spam detection, sentiment analysis, document categorization
- **Швидкий baseline** — перша модель для спроби
- **Малі датасети** — працює навіть з малою кількістю даних
- **Real-time predictions** — дуже швидкі передбачення
- **Multi-class classification** — багато класів
- **Probabilistic outputs** — потрібні ймовірності
- **Online learning** — постійне оновлення моделі
- **High-dimensional sparse data** — text з великим vocabulary

### Краще використати інше ✗

- **Максимальна точність** → Random Forest, XGBoost, SVM
- **Складні взаємодії** → Tree-based, Neural Networks
- **Сильно корельовані ознаки** → інші алгоритми
- **Image/Audio** → Deep Learning
- **Structured tabular data** → Tree-based часто краще

---

## Практичні поради 💡

1. **Почни з Naive Bayes** для text classification — швидкий та ефективний
2. **Multinomial для text** — з TF-IDF або Bag of Words
3. **Gaussian для числових** — перевір нормальність розподілу
4. **Laplace smoothing** — завжди використовуй (α=1)
5. **TF-IDF краще за BoW** — для більшості text задач
6. **Підбирай alpha** — через cross-validation
7. **Візуалізуй top words** — зрозумій, що модель навчила
8. **Порівняй з Logistic Regression** — baseline comparison
9. **Online learning** — легко оновлювати `partial_fit()`
10. **Калібровані ймовірності** — можна довіряти `predict_proba()`

---

## Поширені помилки ❌

### 1. Використовувати на сильно корельованих ознаках

```python
# Якщо ознаки дуже корельовані → порушення незалежності
# Naive Bayes може погано працювати
# ✅ Краще: видали корельовані ознаки або використай інший алгоритм
```

### 2. Забути про smoothing

```python
# ❌ БЕЗ SMOOTHING
mnb = MultinomialNB(alpha=0.0)  # Може бути 0 ймовірності!

# ✅ З LAPLACE SMOOTHING
mnb = MultinomialNB(alpha=1.0)
```

### 3. Використовувати Gaussian NB на non-Gaussian даних

```python
# Якщо дані НЕ нормально розподілені
# Gaussian NB може погано працювати

# ✅ Перевір розподіл:
import matplotlib.pyplot as plt
plt.hist(X[:, 0], bins=30)
plt.show()

# Якщо не нормальний → розглянь інші алгоритми
```

### 4. Не підбирати alpha

```python
# ❌ ПРОСТО ВЗЯТИ α=1.0
mnb = MultinomialNB(alpha=1.0)

# ✅ ПІДІБРАТИ ЧЕРЕЗ CV
from sklearn.model_selection import GridSearchCV
param_grid = {'alpha': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]}
grid = GridSearchCV(MultinomialNB(), param_grid, cv=5)
grid.fit(X_train, y_train)
```

### 5. Ігнорувати class imbalance

```python
# Якщо класи незбалансовані, prior probabilities важливі

# ✅ Можна встановити вручну:
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB(class_prior=[0.3, 0.7])  # Custom priors
```

---

## Online Learning з Naive Bayes

### Incremental Learning

Naive Bayes підтримує **online learning** через `partial_fit()`:

```python
from sklearn.naive_bayes import MultinomialNB

# Ініціалізація
mnb = MultinomialNB()

# Навчання на першому batch
X_batch1, y_batch1 = get_first_batch()
mnb.partial_fit(X_batch1, y_batch1, classes=[0, 1, 2])

# Оновлення на другому batch
X_batch2, y_batch2 = get_second_batch()
mnb.partial_fit(X_batch2, y_batch2)

# Оновлення на третьому batch
X_batch3, y_batch3 = get_third_batch()
mnb.partial_fit(X_batch3, y_batch3)

# Фінальна модель враховує всі батчі!
```

**Переваги:**
- ✅ Не потрібно зберігати всі дані в пам'яті
- ✅ Легко оновлювати модель
- ✅ Real-time learning

**Використання:**
- Stream data
- Very large datasets (не вміщуються в пам'ять)
- Continuously updating systems

---

## Probability Calibration

### Проблема

Naive Bayes часто дає **екстремальні ймовірності** (дуже близькі до 0 або 1).

### Рішення: Calibration

```python
from sklearn.calibration import CalibratedClassifierCV

# Базова модель
mnb = MultinomialNB()

# Calibrated classifier
calibrated_mnb = CalibratedClassifierCV(mnb, cv=5, method='sigmoid')
calibrated_mnb.fit(X_train, y_train)

# Передбачення
y_pred_proba_calibrated = calibrated_mnb.predict_proba(X_test)
```

**Методи калібрації:**
- **Platt scaling** (`method='sigmoid'`)
- **Isotonic regression** (`method='isotonic'`)

---

## Пов'язані теми

- [[02_Logistic_Regression]] — альтернатива для класифікації
- [[04_SVM]] — для high-dimensional text
- [[Text_Classification]] — NLP задачі
- [[Feature_Extraction]] — TF-IDF, Bag of Words
- [[Probability_Theory]] — теорема Байєса

## Ресурси

- [Scikit-learn: Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)
- [Original Paper: "Naive Bayes at Forty"](https://www.cs.unb.ca/~hzhang/publications/FLAIRS04ZhangH.pdf)
- [StatQuest: Naive Bayes](https://www.youtube.com/watch?v=O2L2Uv9pdDA)
- [Text Classification Guide](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)

---

## Ключові висновки

> Naive Bayes — це сімейство швидких probabilistic алгоритмів, що базуються на теоремі Байєса з "наївним" припущенням про незалежність ознак.

**Основні принципи:**
- **Теорема Байєса:** $P(C|X) = \frac{P(X|C) \cdot P(C)}{P(X)}$
- **Naive припущення:** ознаки незалежні → $P(X|C) = \prod P(x_i|C)$
- **Три типи:** Gaussian (continuous), Multinomial (counts), Bernoulli (binary)
- **Laplace smoothing:** запобігає 0 ймовірностям

**Класифікація:**
$$\hat{y} = \arg\max_{C_k} P(C_k) \cdot \prod_{i=1}^{n} P(x_i | C_k)$$

**Типи Naive Bayes:**
- **Gaussian:** для continuous numerical features
- **Multinomial:** для text (word counts) — найпопулярніший!
- **Bernoulli:** для binary features (presence/absence)

**Коли використовувати:**
- Text classification = Multinomial NB ✓
- Швидкий baseline = Naive Bayes ✓
- Real-time predictions = Naive Bayes ✓
- Максимальна точність → інші алгоритми

**КРИТИЧНО важливо:**
- Використовуй Multinomial для text (з TF-IDF)
- Завжди застосовуй Laplace smoothing (α=1)
- Підбирай alpha через cross-validation
- Naive Bayes — чудовий baseline, але не завжди найточніший

**Переваги:**
- Швидкість ⭐⭐⭐⭐⭐
- Простота ⭐⭐⭐⭐⭐
- Text classification ⭐⭐⭐⭐⭐

---

#ml #supervised-learning #classification #naive-bayes #probabilistic #text-classification #bayes-theorem #gaussian #multinomial #bernoulli
