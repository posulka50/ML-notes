# AdaBoost (Adaptive Boosting)

## Що це?

**AdaBoost (Adaptive Boosting)** — це класичний ансамблевий алгоритм boosting, який **адаптивно** змінює ваги тренувальних зразків, фокусуючись на важких для класифікації прикладах.

**Головна ідея:** послідовно навчати слабкі класифікатори (weak learners), даючи більшу вагу зразкам, які були неправильно класифіковані попередніми моделями.

## Навіщо потрібен?

- 🎯 **Простота** — легко зрозуміти та реалізувати
- 📚 **Історична важливість** — перший практичний boosting алгоритм (1996)
- 🔧 **Універсальність** — працює з будь-якими слабкими learners
- 💡 **Інтерпретованість** — зрозуміла логіка вагування
- ⚡ **Ефективність** — може значно покращити weak learners

## Коли використовувати?

**Потрібно:**

- **Бінарна класифікація** — основна задача AdaBoost
- Простий baseline для boosting
- **Навчальні цілі** — розуміння принципів boosting
- Weak learners (decision stumps) доступні
- Дані не дуже зашумлені

**Не потрібно:**
- **Максимальна точність** → Gradient Boosting (XGBoost, LightGBM)
- Регресія → Gradient Boosting Regressor
- **Зашумлені дані** → Random Forest (робастніший)
- Багатокласова класифікація → Gradient Boosting, XGBoost

---

## Відмінність від інших методів

### AdaBoost vs Gradient Boosting

| Характеристика | AdaBoost | Gradient Boosting |
|----------------|----------|-------------------|
| **Підхід** | Змінює ваги зразків | Фітує на residuals |
| **Функція втрат** | Exponential loss | Різні (MSE, Log-loss) |
| **Weak learners** | Будь-які | Зазвичай дерева |
| **Складність** | Простіший | Складніший, гнучкіший |
| **Регресія** | ❌ (складно) | ✅ Так |
| **Популярність** | Історична | ✅ SOTA |

### AdaBoost vs Random Forest

| Характеристика | AdaBoost | Random Forest |
|----------------|----------|---------------|
| **Тип ансамблю** | Boosting (послідовний) | Bagging (паралельний) |
| **Фокус** | Важкі зразки | Різноманітність |
| **Overfitting** | Може переобучитися | Робастний |
| **Швидкість** | Послідовне | Паралельне |

---

## Як працює AdaBoost?

### Інтуїція

**Аналогія: Навчання студента**

1. **Перший тест (модель 1):**
   - Студент вирішує задачі
   - Деякі правильно ✓, деякі неправильно ✗

2. **Другий тест (модель 2):**
   - Викладач **більше уваги** приділяє складним задачам (з помилками)
   - Студент фокусується на важких задачах

3. **Третій тест (модель 3):**
   - Знову фокус на задачах, де були помилки
   - Поступово всі задачі вирішені

**Фінальний іспит:**
- Кожен тест має **вагу** (важливість)
- Тести, де студент краще справився → більша вага
- Фінальна оцінка = зважена комбінація всіх тестів

### Візуалізація

```
Дані: O O O X X X (O = клас 0, X = клас 1)

Модель 1 (decision stump):
    Передбачення: O O X X X X
    Помилки:         ✗       ← помилка
    Дати більшу вагу цьому зразку!

Модель 2 (фокус на важких зразках):
    Передбачення: O O O X X X
    Всі правильно! ✓
    
Фінальне передбачення:
    α₁ * Модель₁ + α₂ * Модель₂
    де α₁, α₂ — ваги моделей
```

---

## Математика

### Алгоритм AdaBoost (SAMME)

**Вхід:**
- Тренувальні дані: $(x_1, y_1), ..., (x_n, y_n)$ де $y_i \in \{-1, +1\}$
- Кількість ітерацій: $T$
- Базовий алгоритм (weak learner)

**1. Ініціалізація ваг:**
$$w_i^{(1)} = \frac{1}{n} \quad \text{для всіх } i = 1, ..., n$$

Всі зразки спочатку мають рівні ваги.

**2. Для $t = 1$ до $T$:**

   **a) Навчити weak learner $h_t$ на даних з вагами $w^{(t)}$**
   
   **b) Обчислити зважену помилку:**
   $$\epsilon_t = \sum_{i=1}^{n} w_i^{(t)} \cdot \mathbb{1}[h_t(x_i) \neq y_i]$$
   
   де $\mathbb{1}[\cdot]$ — індикаторна функція (1 якщо помилка, 0 інакше)

   **c) Обчислити вагу моделі:**
   $$\alpha_t = \frac{1}{2} \ln\left(\frac{1 - \epsilon_t}{\epsilon_t}\right)$$
   
   - Якщо $\epsilon_t$ мала (модель точна) → $\alpha_t$ велика ✓
   - Якщо $\epsilon_t$ велика (модель погана) → $\alpha_t$ мала ✗

   **d) Оновити ваги зразків:**
   $$w_i^{(t+1)} = w_i^{(t)} \cdot \exp(-\alpha_t \cdot y_i \cdot h_t(x_i))$$
   
   або еквівалентно:
   $$w_i^{(t+1)} = w_i^{(t)} \cdot \begin{cases}
   e^{-\alpha_t} & \text{якщо правильно класифіковано} \\
   e^{\alpha_t} & \text{якщо неправильно класифіковано}
   \end{cases}$$
   
   **e) Нормалізувати ваги:**
   $$w_i^{(t+1)} = \frac{w_i^{(t+1)}}{\sum_{j=1}^{n} w_j^{(t+1)}}$$

**3. Фінальна модель (зважене голосування):**
$$H(x) = \text{sign}\left(\sum_{t=1}^{T} \alpha_t \cdot h_t(x)\right)$$

---

## Детальний приклад

### Дані

| № | x₁ | x₂ | y (клас) |
|---|----|----|----------|
| 1 | 1  | 2  | -1 |
| 2 | 2  | 3  | -1 |
| 3 | 3  | 3  | +1 |
| 4 | 4  | 5  | +1 |
| 5 | 5  | 4  | +1 |

### Iteration 1

**a) Ініціалізація ваг:**
$$w^{(1)} = [0.2, 0.2, 0.2, 0.2, 0.2]$$

**b) Навчити weak learner (decision stump):**

Припустимо, найкраще правило: "Якщо x₁ ≤ 2.5, то клас = -1, інакше клас = +1"

**Передбачення:**
| № | Правильний y | Передбачення h₁ | Правильно? |
|---|--------------|-----------------|------------|
| 1 | -1 | -1 | ✓ |
| 2 | -1 | -1 | ✓ |
| 3 | +1 | +1 | ✓ |
| 4 | +1 | +1 | ✓ |
| 5 | +1 | +1 | ✓ |

**c) Обчислити помилку:**
$$\epsilon_1 = 0.2 \times 0 + 0.2 \times 0 + 0.2 \times 0 + 0.2 \times 0 + 0.2 \times 0 = 0$$

⚠️ Помилка = 0 не дозволена! У реальності буде ~0.001

Припустимо, одна помилка (зразок 3):
$$\epsilon_1 = 0.2$$

**d) Вага моделі:**
$$\alpha_1 = \frac{1}{2} \ln\left(\frac{1 - 0.2}{0.2}\right) = \frac{1}{2} \ln(4) \approx 0.693$$

**e) Оновлення ваг зразків:**

Для правильних (наприклад, зразок 1):
$$w_1^{(2)} = 0.2 \times e^{-0.693} = 0.2 \times 0.5 = 0.1$$

Для неправильних (зразок 3):
$$w_3^{(2)} = 0.2 \times e^{0.693} = 0.2 \times 2 = 0.4$$

**f) Нормалізація:**
$$\text{Сума} = 0.1 + 0.1 + 0.4 + 0.1 + 0.1 = 0.8$$

$$w^{(2)} = \left[\frac{0.1}{0.8}, \frac{0.1}{0.8}, \frac{0.4}{0.8}, \frac{0.1}{0.8}, \frac{0.1}{0.8}\right]$$
$$= [0.125, 0.125, 0.5, 0.125, 0.125]$$

**Спостереження:** Зразок 3 тепер має вагу 0.5 (замість 0.2) — більший фокус!

### Iteration 2

**a) Навчити новий weak learner з оновленими вагами $w^{(2)}$**

Модель фокусується на зразку 3 (найбільша вага).

**b-f) Повторити кроки...**

### Фінальне передбачення

Після $T$ ітерацій:

$$H(x) = \text{sign}(\alpha_1 h_1(x) + \alpha_2 h_2(x) + ... + \alpha_T h_T(x))$$

**Приклад:**
- $h_1(x) = +1$, $\alpha_1 = 0.693$
- $h_2(x) = -1$, $\alpha_2 = 0.405$
- $h_3(x) = +1$, $\alpha_3 = 0.916$

$$H(x) = \text{sign}(0.693 \times 1 - 0.405 \times 1 + 0.916 \times 1)$$
$$= \text{sign}(1.204) = +1$$

---

## Вага моделі (α)

### Формула

$$\alpha_t = \frac{1}{2} \ln\left(\frac{1 - \epsilon_t}{\epsilon_t}\right)$$

### Графік α vs ε

```
α (вага моделі)
    |
  3 |                   •
    |                 ╱
  2 |               ╱
    |             ╱
  1 |          •╱
    |        ╱
  0 |______•_______________ ε (помилка)
    0    0.2  0.5  0.8   1.0

При ε = 0.5 (випадкове гадання) → α = 0 (модель непотрібна)
При ε → 0 (ідеальна модель) → α → ∞ (дуже важлива)
При ε → 1 (завжди помиляється) → α → -∞ (негативна вага)
```

**Інтерпретація:**
- **ε < 0.5:** модель краща за випадкове гадання → α > 0 ✓
- **ε = 0.5:** модель = випадкове гадання → α = 0
- **ε > 0.5:** модель гірша за випадкове гадання → α < 0 ✗

---

## Код (scikit-learn)

### Базовий приклад

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Генерація даних
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    random_state=42
)

# Розділення
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# AdaBoost з decision stumps (max_depth=1)
ada_clf = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),  # Decision stump
    n_estimators=50,          # Кількість weak learners
    learning_rate=1.0,        # Швидкість навчання
    algorithm='SAMME',        # або 'SAMME.R' (використовує ймовірності)
    random_state=42
)

# Навчання
ada_clf.fit(X_train, y_train)

# Передбачення
y_pred = ada_clf.predict(X_test)

# Метрики
print("=== AdaBoost Classifier ===")
print(f"Train Accuracy: {ada_clf.score(X_train, y_train):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")

print("\n" + classification_report(y_test, y_pred))

# Ваги моделей (estimator_weights_)
print("\n=== Model Weights (first 10) ===")
print(ada_clf.estimator_weights_[:10])

# Помилки моделей (estimator_errors_)
print("\n=== Model Errors (first 10) ===")
print(ada_clf.estimator_errors_[:10])
```

### З різними weak learners

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# 1. Decision Stumps (найпопулярніше)
ada_stumps = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    random_state=42
)

# 2. Дерева глибиною 3
ada_trees = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=3),
    n_estimators=50,
    random_state=42
)

# 3. Logistic Regression
ada_lr = AdaBoostClassifier(
    estimator=LogisticRegression(max_iter=1000),
    n_estimators=50,
    random_state=42
)

# Порівняння
for name, model in [('Stumps', ada_stumps), 
                     ('Trees', ada_trees), 
                     ('LR', ada_lr)]:
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"{name}: Test Accuracy = {score:.4f}")
```

---

## Staged Predictions

### Моніторинг навчання

```python
# Навчання
ada = AdaBoostClassifier(n_estimators=100, random_state=42)
ada.fit(X_train, y_train)

# Поетапна accuracy
train_scores = []
test_scores = []

for train_pred, test_pred in zip(ada.staged_predict(X_train),
                                   ada.staged_predict(X_test)):
    train_scores.append(accuracy_score(y_train, train_pred))
    test_scores.append(accuracy_score(y_test, test_pred))

# Візуалізація
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(train_scores) + 1), train_scores, 
         label='Train', linewidth=2)
plt.plot(range(1, len(test_scores) + 1), test_scores, 
         label='Test', linewidth=2)
plt.xlabel('Number of Estimators', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('AdaBoost: Accuracy vs Number of Estimators', 
          fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Оптимальна кількість estimators
optimal_n = np.argmax(test_scores) + 1
print(f"Optimal number of estimators: {optimal_n}")
print(f"Best Test Accuracy: {test_scores[optimal_n - 1]:.4f}")
```

---

## Візуалізація Decision Boundaries

```python
from sklearn.datasets import make_moons
import numpy as np

# Генерація даних
X, y = make_moons(n_samples=500, noise=0.3, random_state=42)

# Різна кількість estimators
n_estimators_list = [1, 5, 10, 50]
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

for idx, n_est in enumerate(n_estimators_list):
    # Модель
    ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=n_est,
        random_state=42
    )
    ada.fit(X, y)
    
    # Сітка для візуалізації
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    # Передбачення
    Z = ada.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Візуалізація
    axes[idx].contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    axes[idx].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis',
                     edgecolors='k', s=50)
    axes[idx].set_title(f'n_estimators={n_est}\nAccuracy={ada.score(X, y):.3f}',
                       fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Feature 1')
    axes[idx].set_ylabel('Feature 2')

plt.tight_layout()
plt.show()
```

---

## SAMME vs SAMME.R

### SAMME (Discrete AdaBoost)

**Використовує дискретні передбачення класів:**

$$\alpha_t = \frac{1}{2} \ln\left(\frac{1 - \epsilon_t}{\epsilon_t}\right) + \ln(K - 1)$$

де $K$ — кількість класів.

```python
ada_samme = AdaBoostClassifier(
    algorithm='SAMME',
    n_estimators=50
)
```

### SAMME.R (Real AdaBoost)

**Використовує ймовірності класів:**

Більш гнучкий та зазвичай **точніший**.

```python
ada_sammer = AdaBoostClassifier(
    algorithm='SAMME.R',  # За замовчуванням
    n_estimators=50
)
```

### Порівняння

```python
# SAMME
ada_samme = AdaBoostClassifier(algorithm='SAMME', n_estimators=100)
ada_samme.fit(X_train, y_train)
score_samme = ada_samme.score(X_test, y_test)

# SAMME.R
ada_sammer = AdaBoostClassifier(algorithm='SAMME.R', n_estimators=100)
ada_sammer.fit(X_train, y_train)
score_sammer = ada_sammer.score(X_test, y_test)

print(f"SAMME Accuracy: {score_samme:.4f}")
print(f"SAMME.R Accuracy: {score_sammer:.4f}")
```

**Рекомендація:** Використовуй **SAMME.R** (за замовчуванням) для кращої точності.

---

## Learning Rate

### Вплив на навчання

$$F(x) = \text{sign}\left(\sum_{t=1}^{T} \nu \cdot \alpha_t \cdot h_t(x)\right)$$

де $\nu$ — learning rate.

```python
# Експеримент з learning rate
learning_rates = [0.1, 0.5, 1.0, 2.0]

for lr in learning_rates:
    ada = AdaBoostClassifier(
        n_estimators=100,
        learning_rate=lr,
        random_state=42
    )
    ada.fit(X_train, y_train)
    score = ada.score(X_test, y_test)
    print(f"Learning Rate {lr}: Test Accuracy = {score:.4f}")
```

**Типовий результат:**
```
Learning Rate 0.1: Test Accuracy = 0.8450
Learning Rate 0.5: Test Accuracy = 0.8700
Learning Rate 1.0: Test Accuracy = 0.8750  ← Найкраще
Learning Rate 2.0: Test Accuracy = 0.8600  ← Overfitting
```

**Trade-off:**
- Малий LR → потрібно більше estimators
- Великий LR → ризик overfitting

---

## Переваги та недоліки

### Переваги ✓

| Перевага | Пояснення |
|----------|-----------|
| **Простота** | Легко зрозуміти концепцію |
| **Універсальність** | Працює з різними weak learners |
| **Не потребує tuning** | Працює добре "out-of-the-box" |
| **Історична важливість** | Класичний boosting алгоритм |
| **Автоматичний feature selection** | Фокус на важливих ознаках |
| **Інтерпретованість** | Зрозуміла логіка вагування |

### Недоліки ✗

| Недолік | Пояснення |
|---------|-----------|
| **Чутливість до шуму** | Переобучується на noise та outliers |
| **Чутливість до викидів** | Дає їм велику вагу |
| **Бінарна класифікація** | Основна задача (багатокласова складніша) |
| **Повільніше за RF** | Послідовна природа |
| **Застарілість** | Gradient Boosting зазвичай кращий |
| **Регресія** | Складно адаптувати |

---

## Порівняння з іншими алгоритмами

### AdaBoost vs Gradient Boosting

**Коли AdaBoost:**
✅ Навчальна задача (розуміння boosting)
✅ Простий baseline
✅ Weak learners вже є

**Коли Gradient Boosting:**
✅ **Максимальна точність**
✅ Регресія
✅ Production ML
✅ Більша гнучкість (різні loss functions)

### AdaBoost vs Random Forest

**Коли AdaBoost:**
✅ Послідовна побудова моделей має сенс
✅ Weak learners природно доступні

**Коли Random Forest:**
✅ **Робастність до шуму**
✅ Паралелізація важлива
✅ Менше ризику overfitting
✅ **Рекомендовано для production**

---

## Практичні поради 💡

1. **Decision stumps** — найкраще як weak learners
2. **n_estimators=50-200** — типові значення
3. **SAMME.R** — використовуй за замовчуванням
4. **Staged predictions** — моніторинг overfitting
5. **Обережно з outliers** — AdaBoost дуже чутливий
6. **Очисти дані** — видали шум перед навчанням
7. **Порівняй з Gradient Boosting** — зазвичай точніше
8. **Learning rate=1.0** — хороший старт
9. **Візуалізуй ваги** — розумій, на чому фокусується модель
10. **Не для production** — використовуй XGBoost/LightGBM

---

## Коли використовувати AdaBoost

### Ідеально підходить ✓

- **Навчальні цілі** — розуміння концепції boosting
- Простий baseline для класифікації
- Weak learners вже доступні
- **Історичний контекст** — класичний алгоритм
- Невеликі датасети без шуму

### Краще використати інше ✗

- **Production ML** → Gradient Boosting (XGBoost, LightGBM)
- Максимальна точність → Gradient Boosting
- **Зашумлені дані** → Random Forest
- Регресія → Gradient Boosting Regressor
- Великі дані → LightGBM

---

## Реальний приклад: Детекція спаму

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Синтетичні дані (spam detection)
np.random.seed(42)
n_samples = 2000

# Ознаки email
data = {
    'num_capital_letters': np.random.randint(0, 200, n_samples),
    'num_exclamation': np.random.randint(0, 20, n_samples),
    'num_links': np.random.randint(0, 15, n_samples),
    'word_count': np.random.randint(10, 500, n_samples),
    'num_suspicious_words': np.random.randint(0, 30, n_samples),
    'has_attachment': np.random.randint(0, 2, n_samples),
}

# Симулюємо spam
spam_prob = (
    (data['num_exclamation'] > 10) * 0.3 +
    (data['num_suspicious_words'] > 15) * 0.4 +
    (data['num_capital_letters'] > 100) * 0.2 +
    np.random.uniform(0, 0.1, n_samples)
)
data['is_spam'] = (spam_prob > 0.5).astype(int)

df = pd.DataFrame(data)

X = df.drop('is_spam', axis=1)
y = df['is_spam']

print(f"Dataset shape: {X.shape}")
print(f"Spam rate: {y.mean():.2%}")

# Розділення
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# AdaBoost з decision stumps
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=100,
    learning_rate=1.0,
    algorithm='SAMME.R',
    random_state=42
)

# Навчання
print("\nTraining AdaBoost...")
ada.fit(X_train, y_train)

# Cross-validation
cv_scores = cross_val_score(ada, X_train, y_train, cv=5)
print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# Передбачення
y_pred = ada.predict(X_test)
y_pred_proba = ada.predict_proba(X_test)[:, 1]

# Метрики
print("\n" + "="*60)
print("=== Model Performance ===")
print("="*60)
print(f"Train Accuracy: {ada.score(X_train, y_train):.4f}")
print(f"Test Accuracy: {ada.score(X_test, y_test):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

print("\n" + classification_report(y_test, y_pred, 
                                   target_names=['Not Spam', 'Spam']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Feature Importance (через частоту використання)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': ada.feature_importances_
}).sort_values('importance', ascending=False)

print("\n" + "="*60)
print("=== Top Features ===")
print("="*60)
for idx, row in feature_importance.iterrows():
    print(f"{row['feature']}: {row['importance']:.4f}")

# Візуалізації
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Staged Accuracy
train_staged = []
test_staged = []

for train_pred, test_pred in zip(ada.staged_predict(X_train),
                                   ada.staged_predict(X_test)):
    train_staged.append(accuracy_score(y_train, train_pred))
    test_staged.append(accuracy_score(y_test, test_pred))

axes[0, 0].plot(range(1, len(train_staged) + 1), train_staged, 
                label='Train', linewidth=2)
axes[0, 0].plot(range(1, len(test_staged) + 1), test_staged, 
                label='Test', linewidth=2)
axes[0, 0].set_xlabel('Number of Estimators', fontsize=12)
axes[0, 0].set_ylabel('Accuracy', fontsize=12)
axes[0, 0].set_title('Learning Curve', fontsize=14, fontweight='bold')
axes[0, 0].legend(fontsize=11)
axes[0, 0].grid(True, alpha=0.3)

# 2. Feature Importance
axes[0, 1].barh(feature_importance['feature'], 
                feature_importance['importance'])
axes[0, 1].set_xlabel('Importance', fontsize=12)
axes[0, 1].set_title('Feature Importances', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='x')

# 3. Model Weights Distribution
axes[1, 0].hist(ada.estimator_weights_, bins=20, edgecolor='black')
axes[1, 0].set_xlabel('Model Weight (α)', fontsize=12)
axes[1, 0].set_ylabel('Frequency', fontsize=12)
axes[1, 0].set_title('Distribution of Model Weights', 
                     fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# 4. Error Distribution
axes[1, 1].scatter(range(len(ada.estimator_errors_)), 
                   ada.estimator_errors_, alpha=0.6)
axes[1, 1].axhline(y=0.5, color='r', linestyle='--', 
                   label='Random Guess (ε=0.5)')
axes[1, 1].set_xlabel('Estimator Index', fontsize=12)
axes[1, 1].set_ylabel('Error (ε)', fontsize=12)
axes[1, 1].set_title('Model Errors Over Iterations', 
                     fontsize=14, fontweight='bold')
axes[1, 1].legend(fontsize=11)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## Поширені помилки ❌

### 1. Використання глибоких дерев

```python
# ❌ НЕПРАВИЛЬНО
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=10)
)

# ✅ ПРАВИЛЬНО (decision stumps!)
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1)
)
```

### 2. Не очищати дані від outliers

```python
# AdaBoost дуже чутливий до outliers!
# ✅ Завжди очищуй дані спочатку
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
```

### 3. Використовувати SAMME замість SAMME.R

```python
# ❌ Менш точно
ada = AdaBoostClassifier(algorithm='SAMME')

# ✅ Точніше (за замовчуванням)
ada = AdaBoostClassifier(algorithm='SAMME.R')
```

### 4. Не моніторити overfitting

```python
# ✅ Завжди перевіряй staged predictions
for i, pred in enumerate(ada.staged_predict(X_test)):
    if i % 10 == 0:
        print(f"After {i+1} estimators: {accuracy_score(y_test, pred):.4f}")
```

---

## Пов'язані теми

- [[01_Decision_Trees]] — weak learners
- [[03_Gradient_Boosting]] — сучасніша альтернатива
- [[05_Ensemble_Methods]] — теорія ансамблів
- [[02_Random_Forest]] — bagging vs boosting

## Ресурси

- [Scikit-learn: AdaBoost](https://scikit-learn.org/stable/modules/ensemble.html#adaboost)
- [Original Paper: Freund & Schapire (1997)](https://www.sciencedirect.com/science/article/pii/S002200009791504X)
- [StatQuest: AdaBoost](https://www.youtube.com/watch?v=LsK-xG1cLYA)

---

## Ключові висновки

> AdaBoost послідовно навчає weak learners, адаптивно збільшуючи ваги важких для класифікації зразків та комбінуючи моделі через зважене голосування.

**Основні принципи:**
- **Adaptive re-weighting** — фокус на важких зразках
- **Weak learners** — зазвичай decision stumps (max_depth=1)
- **Зважене голосування** — кращі моделі мають більшу вагу
- **Exponential loss** — функція втрат

**Формула ваги моделі:**
$$\alpha_t = \frac{1}{2} \ln\left(\frac{1 - \epsilon_t}{\epsilon_t}\right)$$

**Коли використовувати:**
- Навчання концепції boosting = AdaBoost ✓
- Production ML = Gradient Boosting (XGBoost) ✓
- Робастність до шуму = Random Forest ✓

**Найважливіше:**
- Decision stumps як weak learners
- SAMME.R за замовчуванням
- Обережно з outliers та шумом
- Для production використовуй XGBoost/LightGBM

---

#ml #supervised-learning #ensemble #adaboost #boosting #classification #weak-learners #tree-based
