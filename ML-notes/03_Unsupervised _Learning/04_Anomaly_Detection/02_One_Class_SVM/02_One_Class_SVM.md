# One-Class SVM

## Що це?

**One-Class SVM** — алгоритм для виявлення аномалій, який навчається **тільки на нормальних даних**. Він будує границю навколо нормальних прикладів у просторі ознак. Все що потрапляє за цю границю вважається аномалією.

**Основна ідея:** навчити модель розпізнавати "норму", щоб потім виявляти все що не вписується в цю норму.

---

## Навіщо потрібен?

**Класична проблема:** у тебе є тисячі нормальних прикладів і мало (або зовсім немає) аномалій для навчання.

**Типові застосування:**

1. **Виробництво** — виявлення бракованих деталей коли є тисячі нормальних і купка дефектних
2. **Кібербезпека** — детекція атак коли є мільйони нормальних запитів і рідкісні атаки
3. **Медицина** — виявлення рідкісних захворювань коли здорових пацієнтів багато
4. **Фінанси** — виявлення шахрайства коли більшість транзакцій чесні
5. **Моніторинг обладнання** — виявлення несправностей до поломки

**Коли це працює:** маєш багато нормальних даних, мало або зовсім немає аномалій, і неможливо передбачити всі типи аномалій наперед.

---

## Як це працює?

**Процес:**

1. **Навчання** — алгоритм бере всі нормальні дані і будує навколо них границю у просторі ознак
2. **Prediction** — нові точки перевіряються: всередині границі = нормально, ззовні = аномалія

**Kernel trick** — можливість будувати складні, нелінійні границі замість простих прямих ліній. Це дозволяє відокремлювати складні паттерни.

**Типи kernel:**

- **Linear** — прямі лінії, для простих випадків
- **RBF (Gaussian)** — найпопулярніший, дозволяє складні форми, використовуй за замовчуванням
- **Polynomial** — поліноміальні криві, рідко потрібен
- **Sigmoid** — S-подібні криві, специфічні випадки

## Ключові параметри

### nu (ню)

Контролює **строгість** boundaries. Це upper bound на частку outliers в тренувальних даних.

**Значення:**

- `nu=0.01` (1%) — дуже строго, тільки 1% може бути за межею
- `nu=0.1` (10%) — стандартно, підходить для більшості випадків
- `nu=0.5` (50%) — м'яко, дозволяє багато винятків

**Коли що:**

- Критичні системи (медицина, фінанси) → низький nu (0.01-0.05)
- Звичайне використання → середній nu (0.1)
- Шумні дані → високий nu (0.2-0.5)

**Проблеми:**

- Багато false positives → збільш nu
- Пропускає аномалії → зменш nu

### gamma (гамма)

Контролює **складність** boundaries для RBF kernel. Визначає радіус впливу кожної training точки.

**Значення:**

- `gamma=0.001` — широкий вплив, гладкі границі
- `gamma=0.1` — збалансовано
- `gamma=1.0` — вузький вплив, складні границі (ризик overfitting)

**Коли що:**

- Простий pattern → низький gamma
- Складний pattern → високий gamma
- Не впевнений → `gamma='auto'` (автоматично підбере)

**Проблеми:**

- Границя занадто проста → збільш gamma
- Границя занадто складна (overfit) → зменш gamma

### Інші параметри

```python
OneClassSVM(
    kernel='rbf',      # Тип kernel: 'linear', 'rbf', 'poly', 'sigmoid'
    gamma='scale',     # Коефіцієнт для RBF
    nu=0.5,            # Upper bound на outliers
    degree=3,          # Степінь для poly kernel
    tol=1e-3,          # Точність для зупинки
    max_iter=-1        # Максимум ітерацій (-1 = без ліміту)
)
```

---

## Приклад використання

### Базовий приклад: Виявлення дефектів

```python
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import numpy as np

# 1. Нормальні дані для навчання
normal_data = np.random.randn(200, 5)  # 200 нормальних зразків, 5 features

# 2. ОБОВ'ЯЗКОВО нормалізувати
scaler = StandardScaler()
X_scaled = scaler.fit_transform(normal_data)

# 3. Створити і навчити модель
clf = OneClassSVM(
    kernel='rbf',
    gamma='auto',
    nu=0.1
)
clf.fit(X_scaled)

# 4. Перевірити нові дані
new_sample = [[0.5, -0.3, 0.8, 0.1, -0.5]]
new_scaled = scaler.transform(new_sample)

prediction = clf.predict(new_scaled)[0]
score = clf.decision_function(new_scaled)[0]

if prediction == -1:
    print(f"Аномалія! Score: {score:.3f}")
else:
    print(f"Нормально. Score: {score:.3f}")
```

### Реальний кейс: Виявлення мережевих атак

```python
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Генерація нормального трафіку
np.random.seed(42)
n_normal = 1000

# Features: розмір пакету, тривалість з'єднання, запитів за секунду
packet_size = np.random.normal(500, 100, n_normal)
duration = np.random.exponential(2, n_normal)
requests_per_sec = np.random.poisson(10, n_normal)

X_train = np.column_stack([packet_size, duration, requests_per_sec])

# Тестові дані: нормальні + атаки
n_test_normal = 200
n_attacks = 50

# Нормальні
test_normal = np.column_stack([
    np.random.normal(500, 100, n_test_normal),
    np.random.exponential(2, n_test_normal),
    np.random.poisson(10, n_test_normal)
])

# DDoS атака: малі пакети, багато запитів
attack_ddos = np.column_stack([
    np.random.normal(100, 20, n_attacks//2),
    np.random.uniform(0.1, 0.5, n_attacks//2),
    np.random.poisson(100, n_attacks//2)
])

# Exfiltration: великі пакети
attack_exfil = np.column_stack([
    np.random.normal(5000, 500, n_attacks//2),
    np.random.uniform(10, 20, n_attacks//2),
    np.random.poisson(5, n_attacks//2)
])

X_test = np.vstack([test_normal, attack_ddos, attack_exfil])
y_test = np.array([1]*n_test_normal + [-1]*n_attacks)

# Нормалізація
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Навчання
clf = OneClassSVM(kernel='rbf', gamma=0.1, nu=0.05)
clf.fit(X_train_scaled)

# Prediction
y_pred = clf.predict(X_test_scaled)

# Результати
print("\nРезультати виявлення атак:")
print(classification_report(y_test, y_pred, target_names=['Атака', 'Норма']))

# Аналіз найбільш підозрілих
scores = clf.decision_function(X_test_scaled)
most_suspicious = np.argsort(scores)[:10]

print("\nНайбільш підозрілі зразки:")
for idx in most_suspicious:
    label = "АТАКА" if y_test[idx] == -1 else "Норма"
    print(f"  #{idx}: score={scores[idx]:.3f} | True: {label}")
```

## Переваги і недоліки

### Переваги

|Перевага|Пояснення|
|---|---|
|**Kernel trick**|Може будувати складні нелінійні boundaries|
|**Novelty detection**|Виявляє нові типи аномалій, які не бачив раніше|
|**Математично обґрунтований**|Є теоретична база, передбачувана поведінка|
|**Гладкі границі**|Менш чутливий до noise порівняно з деякими іншими методами|
|**Мало параметрів**|Тільки nu і gamma, легко налаштувати|

### Недоліки

|Недолік|Пояснення|
|---|---|
|**Повільний**|O(n²) до O(n³), не масштабується|
|**Великі дані**|Погано працює на датасетах >10K зразків|
|**Memory intensive**|Споживає багато пам'яті|
|**Потребує scaling**|Обов'язкова нормалізація даних|
|**Parameter tuning**|Треба підбирати nu і gamma|

### Коли використовувати

**Використовуй One-Class SVM:**

- Малі-середні датасети (<10K зразків)
- Складні нелінійні patterns
- Є час на підбір параметрів
- Важлива точність більше за швидкість

**Використовуй Isolation Forest:**

- Великі датасети (>10K зразків)
- Потрібна швидкість
- Не хочеш налаштовувати параметри
- Треба пояснювати результати

## Практичні поради

### Базовий темплейт

```python
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

# 1. Нормалізація
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 2. Модель
clf = OneClassSVM(kernel='rbf', gamma='auto', nu=0.1)
clf.fit(X_train_scaled)

# 3. Prediction
X_test_scaled = scaler.transform(X_test)
predictions = clf.predict(X_test_scaled)
scores = clf.decision_function(X_test_scaled)
```

### Підбір параметрів

```python
best_score = -1
best_params = None

for nu in [0.01, 0.05, 0.1, 0.2]:
    for gamma in [0.001, 0.01, 0.1, 1.0]:
        clf = OneClassSVM(nu=nu, gamma=gamma)
        clf.fit(X_train_scaled)
        
        score = evaluate_on_validation(clf, X_val_scaled)
        
        if score > best_score:
            best_score = score
            best_params = {'nu': nu, 'gamma': gamma}
```

### Ensemble підхід

```python
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

# Навчити обидва алгоритми
ocsvm = OneClassSVM(nu=0.1)
ocsvm.fit(X_train)

iforest = IsolationForest(contamination=0.1)
iforest.fit(X_train)

# Отримати scores
ocsvm_scores = ocsvm.decision_function(X_test)
if_scores = iforest.score_samples(X_test)

# Нормалізувати до [0, 1]
scaler = MinMaxScaler()
ocsvm_norm = scaler.fit_transform(ocsvm_scores.reshape(-1, 1)).ravel()
if_norm = scaler.fit_transform(if_scores.reshape(-1, 1)).ravel()

# Комбінувати
combined = (ocsvm_norm + if_norm) / 2
```

### Streaming detection

```python
class StreamingOCSVM:
    def __init__(self, window_size=1000, nu=0.1):
        self.window_size = window_size
        self.clf = OneClassSVM(nu=nu)
        self.buffer = []
        self.scaler = StandardScaler()
    
    def add_sample(self, x):
        self.buffer.append(x)
        
        if len(self.buffer) >= self.window_size:
            X = np.array(self.buffer[-self.window_size:])
            X_scaled = self.scaler.fit_transform(X)
            self.clf.fit(X_scaled)
            
            self.buffer = self.buffer[-self.window_size//2:]
    
    def predict(self, x):
        x_scaled = self.scaler.transform([x])
        return self.clf.predict(x_scaled)[0]
```

## Порівняння з Isolation Forest

|Критерій|One-Class SVM|Isolation Forest|
|---|---|---|
|**Швидкість навчання**|Повільно (O(n²-n³))|Швидко (O(n log n))|
|**Масштабованість**|До 10K зразків|Мільйони зразків|
|**Boundaries**|Складні нелінійні|Прямолінійні|
|**Параметри**|Треба підбирати|Майже не треба|
|**Інтерпретація**|Складно|Легко|
|**Memory**|Багато|Мало|

**Рекомендація:**

- Дані <10K + складні patterns → One-Class SVM
- Дані >10K або потрібна швидкість → Isolation Forest

## Підсумок

**One-Class SVM** — алгоритм для anomaly detection що навчається тільки на нормальних даних. Використовує kernel trick для побудови складних нелінійних boundaries.

**Основні характеристики:**

- Навчання тільки на нормальних даних
- Kernel trick для складних форм
- Два ключові параметри: nu (строгість) і gamma (гнучкість)
- Обов'язкова нормалізація даних
- Працює для малих-середніх датасетів (<10K)

**Коли використовувати:**

- Є багато нормальних даних, мало аномалій
- Patterns складні і нелінійні
- Датасет малий-середній (<10K зразків)
- Є час на підбір параметрів

**Коли НЕ використовувати:**

- Великі датасети (>10K) → використай Isolation Forest
- Потрібна швидкість → використай Isolation Forest
- Треба пояснювати результати → використай Isolation Forest

**Базовий workflow:**

1. Нормалізуй дані (StandardScaler)
2. Почни з `kernel='rbf'`, `gamma='auto'`, `nu=0.1`
3. Grid search для nu і gamma якщо потрібно
4. Використовуй decision_function для ranking

---

#machine-learning #anomaly-detection #one-class-svm #unsupervised-learning