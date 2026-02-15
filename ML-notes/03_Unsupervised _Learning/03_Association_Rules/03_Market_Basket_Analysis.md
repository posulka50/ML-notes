# Market Basket Analysis

## Що це?

**Market Basket Analysis (MBA)** — це техніка data mining для виявлення **асоціацій між товарами**, які клієнти купують разом. Це практичне застосування association rules mining (Apriori, FP-Growth) у retail та e-commerce.

**Головна ідея:** знайти паттерни типу "клієнти, що купують X, також купують Y", щоб оптимізувати продажі, розташування товарів, рекомендації.

## Навіщо потрібен?

- 🛒 **Product placement** — де розміщувати товари в магазині
- 💰 **Cross-selling** — які товари пропонувати разом
- 🎯 **Bundling** — які bundle пропозиції створювати
- 📧 **Personalized marketing** — targeted campaigns
- 📊 **Inventory management** — оптимізація запасів
- 🏪 **Store layout** — планування розташування відділів
- 💡 **Product recommendations** — "Часто купують разом"

## Бізнес-цінність

### Типові результати

**Приклад 1: Diapers & Beer**
```
Правило: {Diapers} → {Beer}
Support: 15%
Confidence: 75%
Lift: 2.3

Інтерпретація:
- 15% транзакцій містять обидва товари
- 75% хто купує підгузки також купує пиво
- Lift 2.3 → в 2.3 рази більша ймовірність ніж випадково

Бізнес дія:
✓ Розмістити пиво поряд з дитячими товарами
✓ Bundle пропозиція: "Pampers + Budweiser"
✓ Email campaign: "Bought diapers? Try our beer selection"
```

**Приклад 2: Bread & Butter**
```
Правило: {Bread} → {Butter}
Support: 20%
Confidence: 60%
Lift: 1.5

Дія:
✓ Розмістити поруч в магазині
✓ Знижка на butter при купівлі bread
✓ Recipe suggestions (Bread + Butter recipes)
```

---

## Повний бізнес-процес MBA

### 1. Бізнес-питання

**Типові питання:**
- Які товари купують разом?
- Як оптимізувати розташування товарів?
- Які bundle пропозиції створити?
- Кому відправити targeted marketing?
- Як збільити average basket size?

### 2. Збір даних

**Джерела:**
- POS (Point of Sale) systems
- E-commerce transaction logs
- Loyalty card data
- Online clickstream data

**Формат:**
```
Transaction_ID, Date, Customer_ID, Products
T001, 2024-01-15, C123, [Milk, Bread, Butter]
T002, 2024-01-15, C456, [Beer, Chips, Diapers]
T003, 2024-01-15, C789, [Coffee, Sugar, Milk]
```

### 3. Data preprocessing

```python
import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

# Завантажити дані
df_raw = pd.read_csv('transactions.csv')

print(f"Total transactions: {len(df_raw)}")
print(f"Date range: {df_raw['Date'].min()} to {df_raw['Date'].max()}")
print(f"Unique customers: {df_raw['Customer_ID'].nunique()}")

# Перетворити в transaction format
transactions = df_raw.groupby('Transaction_ID')['Product'].apply(list).values.tolist()

print(f"\nSample transactions:")
for i, trans in enumerate(transactions[:5], 1):
    print(f"T{i}: {trans}")

# Очистити дані
def clean_transactions(transactions):
    """Очистити транзакції"""
    clean = []
    
    for trans in transactions:
        # Remove nulls, empty strings
        trans_clean = [item.strip() for item in trans if item and str(item).strip()]
        
        # Remove duplicates in same transaction
        trans_clean = list(set(trans_clean))
        
        # Keep only non-empty
        if len(trans_clean) > 0:
            clean.append(trans_clean)
    
    return clean

transactions_clean = clean_transactions(transactions)

print(f"\nAfter cleaning: {len(transactions_clean)} transactions")

# Статистика
transaction_sizes = [len(t) for t in transactions_clean]
print(f"\nTransaction size stats:")
print(f"  Mean: {np.mean(transaction_sizes):.2f}")
print(f"  Median: {np.median(transaction_sizes):.0f}")
print(f"  Max: {max(transaction_sizes)}")
print(f"  Min: {min(transaction_sizes)}")
```

### 4. Exploratory Data Analysis

```python
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Топ товарів
all_items = [item for trans in transactions_clean for item in trans]
item_counts = Counter(all_items)

top_items = item_counts.most_common(20)

plt.figure(figsize=(12, 6))
items, counts = zip(*top_items)
plt.barh(range(len(items)), counts)
plt.yticks(range(len(items)), items)
plt.xlabel('Frequency', fontsize=12)
plt.title('Top 20 Most Frequent Items', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

print("\n=== Top 10 Items ===")
for item, count in top_items[:10]:
    pct = count / len(transactions_clean) * 100
    print(f"{item:20s}: {count:5d} ({pct:5.2f}%)")

# Розподіл розміру транзакцій
plt.figure(figsize=(10, 6))
plt.hist(transaction_sizes, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Transaction Size (number of items)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of Transaction Sizes', fontsize=14, fontweight='bold')
plt.axvline(np.mean(transaction_sizes), color='red', 
           linestyle='--', label=f'Mean: {np.mean(transaction_sizes):.1f}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Temporal analysis
df_raw['Date'] = pd.to_datetime(df_raw['Date'])
df_raw['DayOfWeek'] = df_raw['Date'].dt.day_name()
df_raw['Hour'] = df_raw['Date'].dt.hour

# Транзакції по днях тижня
plt.figure(figsize=(10, 6))
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
            'Friday', 'Saturday', 'Sunday']
day_counts = df_raw.groupby('DayOfWeek').size().reindex(day_order)
day_counts.plot(kind='bar', color='steelblue', edgecolor='black')
plt.xlabel('Day of Week', fontsize=12)
plt.ylabel('Number of Transactions', fontsize=12)
plt.title('Transactions by Day of Week', fontsize=14, fontweight='bold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### 5. Association Rules Mining

```python
# Transform до binary matrix
te = TransactionEncoder()
te_ary = te.fit(transactions_clean).transform(transactions_clean)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

print(f"Binary matrix shape: {df_encoded.shape}")

# FP-Growth (швидше на великих даних)
print("\nMining frequent itemsets...")
frequent_itemsets = fpgrowth(
    df_encoded,
    min_support=0.01,  # 1% - adjust based on data
    use_colnames=True
)

print(f"Found {len(frequent_itemsets)} frequent itemsets")

# Association rules
print("\nGenerating association rules...")
rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.3  # 30%
)

# Додаткові метрики
rules['antecedent_len'] = rules['antecedents'].apply(lambda x: len(x))
rules['consequent_len'] = rules['consequents'].apply(lambda x: len(x))

print(f"Generated {len(rules)} rules")
print()

# Фільтрувати strong rules
strong_rules = rules[
    (rules['confidence'] >= 0.5) &  # 50%+
    (rules['lift'] >= 1.2) &         # 20%+ lift
    (rules['support'] >= 0.01)       # 1%+ support
]

print(f"Strong rules: {len(strong_rules)}")
```

### 6. Інтерпретація та Insights

```python
# Топ правила за lift
print("\n=== Top 10 Rules by Lift ===\n")
top_rules = rules.nlargest(10, 'lift')

for idx, rule in top_rules.iterrows():
    ant = ', '.join(list(rule['antecedents']))
    cons = ', '.join(list(rule['consequents']))
    
    print(f"{ant} → {cons}")
    print(f"  Support: {rule['support']:.3f} ({rule['support']*100:.1f}%)")
    print(f"  Confidence: {rule['confidence']:.3f} ({rule['confidence']*100:.1f}%)")
    print(f"  Lift: {rule['lift']:.3f}")
    print()

# Групувати за категоріями (якщо є)
def categorize_item(item):
    """Категоризувати товар"""
    # Приклад логіки
    if any(word in item.lower() for word in ['milk', 'cheese', 'butter', 'yogurt']):
        return 'Dairy'
    elif any(word in item.lower() for word in ['bread', 'buns', 'bagel']):
        return 'Bakery'
    elif any(word in item.lower() for word in ['beer', 'wine', 'vodka']):
        return 'Alcohol'
    # ... more categories
    else:
        return 'Other'

# Аналіз cross-category rules
def get_category_cross_rules(rules):
    """Знайти правила між категоріями"""
    cross_rules = []
    
    for _, rule in rules.iterrows():
        ant_items = list(rule['antecedents'])
        cons_items = list(rule['consequents'])
        
        ant_cats = set(categorize_item(item) for item in ant_items)
        cons_cats = set(categorize_item(item) for item in cons_items)
        
        if ant_cats != cons_cats:  # Different categories
            cross_rules.append({
                'ant_category': ant_cats,
                'cons_category': cons_cats,
                'rule': rule
            })
    
    return cross_rules

cross_category = get_category_cross_rules(strong_rules)
print(f"\n=== Cross-Category Rules: {len(cross_category)} ===")
```

### 7. Візуалізація

```python
# Network graph
import networkx as nx

def plot_rules_network(rules, top_n=15):
    """Network graph сильних правил"""
    
    top_rules = rules.nlargest(top_n, 'lift')
    
    G = nx.DiGraph()
    
    for _, rule in top_rules.iterrows():
        for ant in rule['antecedents']:
            for cons in rule['consequents']:
                G.add_edge(
                    ant, cons,
                    weight=rule['lift'],
                    confidence=rule['confidence']
                )
    
    plt.figure(figsize=(14, 10))
    
    pos = nx.spring_layout(G, k=1.5, iterations=50)
    
    # Nodes
    nx.draw_networkx_nodes(G, pos, node_size=3000,
                          node_color='lightblue',
                          alpha=0.8, edgecolors='black', linewidths=2)
    
    # Labels
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')
    
    # Edges
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    
    nx.draw_networkx_edges(
        G, pos,
        width=[w*2 for w in weights],
        alpha=0.5,
        edge_color='gray',
        arrows=True,
        arrowsize=20,
        arrowstyle='->',
        connectionstyle='arc3,rad=0.1'
    )
    
    plt.title('Product Association Network\n(Edge width = Lift)', 
             fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

plot_rules_network(strong_rules)

# Scatter plot
plt.figure(figsize=(12, 8))

scatter = plt.scatter(
    rules['support'],
    rules['confidence'],
    s=rules['lift'] * 100,
    alpha=0.6,
    c=rules['lift'],
    cmap='viridis',
    edgecolors='black',
    linewidths=0.5
)

plt.colorbar(scatter, label='Lift')
plt.xlabel('Support', fontsize=12)
plt.ylabel('Confidence', fontsize=12)
plt.title('Association Rules: Support vs Confidence\n(Size and Color = Lift)',
         fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# Annotate top rules
top_5 = rules.nlargest(5, 'lift')
for _, rule in top_5.iterrows():
    ant = ', '.join(list(rule['antecedents']))[:15]
    cons = ', '.join(list(rule['consequents']))[:15]
    plt.annotate(
        f"{ant}→{cons}",
        (rule['support'], rule['confidence']),
        fontsize=7,
        alpha=0.7
    )

plt.tight_layout()
plt.show()

# Heatmap категорій
def plot_category_heatmap(rules):
    """Heatmap асоціацій між категоріями"""
    
    # Категорії
    categories = set()
    for _, rule in rules.iterrows():
        for item in rule['antecedents']:
            categories.add(categorize_item(item))
        for item in rule['consequents']:
            categories.add(categorize_item(item))
    
    categories = sorted(list(categories))
    
    # Матриця
    matrix = np.zeros((len(categories), len(categories)))
    
    for _, rule in rules.iterrows():
        for ant in rule['antecedents']:
            for cons in rule['consequents']:
                ant_cat = categorize_item(ant)
                cons_cat = categorize_item(cons)
                
                i = categories.index(ant_cat)
                j = categories.index(cons_cat)
                
                matrix[i, j] += rule['lift']
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt='.1f',
        cmap='YlOrRd',
        xticklabels=categories,
        yticklabels=categories,
        cbar_kws={'label': 'Total Lift'}
    )
    plt.xlabel('Consequent Category', fontsize=12)
    plt.ylabel('Antecedent Category', fontsize=12)
    plt.title('Category Cross-Selling Heatmap', 
             fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

plot_category_heatmap(strong_rules)
```

### 8. Бізнес рекомендації

```python
def generate_recommendations(rules, top_n=20):
    """Генерувати бізнес рекомендації"""
    
    recommendations = []
    
    top_rules = rules.nlargest(top_n, 'lift')
    
    for _, rule in top_rules.iterrows():
        ant = ', '.join(list(rule['antecedents']))
        cons = ', '.join(list(rule['consequents']))
        
        # Store layout
        rec = {
            'type': 'Store Layout',
            'action': f"Place '{cons}' near '{ant}'",
            'reason': f"Lift: {rule['lift']:.2f}, Confidence: {rule['confidence']:.1%}",
            'expected_impact': f"{rule['support']*100:.1f}% of transactions affected"
        }
        recommendations.append(rec)
        
        # Bundle
        if rule['lift'] > 1.5 and rule['confidence'] > 0.6:
            rec = {
                'type': 'Bundle Offer',
                'action': f"Create bundle: '{ant}' + '{cons}'",
                'reason': f"Strong association (lift={rule['lift']:.2f})",
                'expected_impact': f"Potential {rule['confidence']*100:.0f}% conversion"
            }
            recommendations.append(rec)
        
        # Marketing campaign
        if rule['support'] > 0.05:  # High support
            rec = {
                'type': 'Email Campaign',
                'action': f"Send '{cons}' offer to customers who bought '{ant}'",
                'reason': f"{rule['confidence']*100:.0f}% likely to buy",
                'expected_impact': f"Target {rule['support']*100:.1f}% of customer base"
            }
            recommendations.append(rec)
    
    return recommendations

recs = generate_recommendations(strong_rules)

print("\n=== Business Recommendations ===\n")
for i, rec in enumerate(recs[:10], 1):
    print(f"{i}. [{rec['type']}] {rec['action']}")
    print(f"   Reason: {rec['reason']}")
    print(f"   Impact: {rec['expected_impact']}")
    print()
```

---

## Сегментація клієнтів

### RFM Analysis + MBA

```python
# Розрахувати RFM метрики
def calculate_rfm(df):
    """Recency, Frequency, Monetary"""
    
    current_date = df['Date'].max() + pd.Timedelta(days=1)
    
    rfm = df.groupby('Customer_ID').agg({
        'Date': lambda x: (current_date - x.max()).days,  # Recency
        'Transaction_ID': 'count',                         # Frequency
        'Amount': 'sum'                                    # Monetary
    })
    
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    
    return rfm

rfm = calculate_rfm(df_raw)

# Сегментувати
def segment_customers(rfm):
    """Сегментація за RFM"""
    
    # Quartiles
    r_labels = range(4, 0, -1)  # 4=best (recent), 1=worst
    f_labels = range(1, 5)       # 4=best (frequent), 1=worst
    m_labels = range(1, 5)
    
    rfm['R_Score'] = pd.qcut(rfm['Recency'], 4, labels=r_labels)
    rfm['F_Score'] = pd.qcut(rfm['Frequency'], 4, labels=f_labels)
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], 4, labels=m_labels)
    
    rfm['RFM_Score'] = (
        rfm['R_Score'].astype(str) +
        rfm['F_Score'].astype(str) +
        rfm['M_Score'].astype(str)
    )
    
    # Сегменти
    def assign_segment(row):
        if row['R_Score'] >= 3 and row['F_Score'] >= 3 and row['M_Score'] >= 3:
            return 'Champions'
        elif row['R_Score'] >= 3 and row['F_Score'] >= 2:
            return 'Loyal Customers'
        elif row['R_Score'] >= 3:
            return 'Potential Loyalists'
        elif row['F_Score'] <= 2 and row['R_Score'] <= 2:
            return 'At Risk'
        elif row['R_Score'] <= 2:
            return 'Lost'
        else:
            return 'Others'
    
    rfm['Segment'] = rfm.apply(assign_segment, axis=1)
    
    return rfm

rfm_segments = segment_customers(rfm)

print("=== Customer Segments ===")
print(rfm_segments['Segment'].value_counts())

# MBA для кожного сегмента окремо
def mba_by_segment(df, segment_name):
    """Association rules для конкретного сегмента"""
    
    # Фільтрувати customers
    segment_customers = rfm_segments[
        rfm_segments['Segment'] == segment_name
    ].index
    
    segment_trans = df[
        df['Customer_ID'].isin(segment_customers)
    ]
    
    # MBA
    transactions = segment_trans.groupby('Transaction_ID')['Product'].apply(list).values.tolist()
    
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    
    frequent = fpgrowth(df_encoded, min_support=0.05, use_colnames=True)
    rules = association_rules(frequent, metric="confidence", min_threshold=0.4)
    
    return rules

# Правила для Champions
rules_champions = mba_by_segment(df_raw, 'Champions')
print(f"\nChampions: {len(rules_champions)} rules")

# Правила для At Risk
rules_atrisk = mba_by_segment(df_raw, 'At Risk')
print(f"At Risk: {len(rules_atrisk)} rules")

# Персоналізовані рекомендації за сегментом!
```

---

## A/B Testing

### Валідація рекомендацій

```python
# Експеримент: перевірити чи працює recommendation

# 1. Базовий період (до зміни)
baseline_data = df_raw[df_raw['Date'] < '2024-02-01']

# 2. Застосувати рекомендацію
# Наприклад: розмістити Beer поряд з Diapers

# 3. Тестовий період (після зміни)
test_data = df_raw[df_raw['Date'] >= '2024-02-01']

# 4. Виміряти ефект
def measure_impact(baseline, test, itemA, itemB):
    """Виміряти зміну co-purchase rate"""
    
    def copurchase_rate(data, itemA, itemB):
        """% транзакцій з обома items"""
        trans = data.groupby('Transaction_ID')['Product'].apply(set)
        
        both = sum(1 for t in trans if itemA in t and itemB in t)
        total = len(trans)
        
        return both / total
    
    baseline_rate = copurchase_rate(baseline, itemA, itemB)
    test_rate = copurchase_rate(test, itemA, itemB)
    
    lift = (test_rate - baseline_rate) / baseline_rate
    
    print(f"Co-purchase rate: {itemA} & {itemB}")
    print(f"  Baseline: {baseline_rate:.2%}")
    print(f"  Test: {test_rate:.2%}")
    print(f"  Lift: {lift:+.1%}")
    
    return lift

lift = measure_impact(baseline_data, test_data, 'Diapers', 'Beer')

# Статистична значущість
from scipy.stats import chi2_contingency

def test_significance(baseline, test, itemA, itemB):
    """Chi-square test"""
    
    def get_contingency_table(data, itemA, itemB):
        trans = data.groupby('Transaction_ID')['Product'].apply(set)
        
        both = sum(1 for t in trans if itemA in t and itemB in t)
        only_a = sum(1 for t in trans if itemA in t and itemB not in t)
        only_b = sum(1 for t in trans if itemB in t and itemA not in t)
        neither = len(trans) - both - only_a - only_b
        
        return [[both, only_a], [only_b, neither]]
    
    table_baseline = get_contingency_table(baseline, itemA, itemB)
    table_test = get_contingency_table(test, itemA, itemB)
    
    # Chi-square
    _, p_value, _, _ = chi2_contingency(table_baseline + table_test)
    
    print(f"\nSignificance test: p-value = {p_value:.4f}")
    
    if p_value < 0.05:
        print("✓ Statistically significant!")
    else:
        print("✗ Not significant")
    
    return p_value

test_significance(baseline_data, test_data, 'Diapers', 'Beer')
```

---

## ROI калькуляція

### Оцінка бізнес-впливу

```python
def calculate_roi(rule, avg_transaction_value, implementation_cost):
    """ROI рекомендації"""
    
    # Estimate
    affected_transactions = rule['support'] * total_transactions
    conversion_rate = rule['confidence']
    
    # Потенційна додаткова виручка
    additional_sales = affected_transactions * conversion_rate * avg_transaction_value
    
    # ROI
    roi = (additional_sales - implementation_cost) / implementation_cost
    
    return {
        'affected_transactions': affected_transactions,
        'additional_sales': additional_sales,
        'implementation_cost': implementation_cost,
        'net_profit': additional_sales - implementation_cost,
        'roi': roi
    }

# Приклад
top_rule = strong_rules.iloc[0]

avg_transaction_value = 50  # $50
implementation_cost = 5000   # $5000 (store reorganization)
total_transactions = 100000  # за місяць

roi_calc = calculate_roi(top_rule, avg_transaction_value, implementation_cost)

print("\n=== ROI Calculation ===")
print(f"Rule: {list(top_rule['antecedents'])} → {list(top_rule['consequents'])}")
print(f"Affected transactions: {roi_calc['affected_transactions']:.0f}")
print(f"Additional sales: ${roi_calc['additional_sales']:,.0f}")
print(f"Implementation cost: ${roi_calc['implementation_cost']:,.0f}")
print(f"Net profit: ${roi_calc['net_profit']:,.0f}")
print(f"ROI: {roi_calc['roi']*100:.1f}%")
```

---

## Практичні поради 💡

### 1. Почни з exploratory analysis

```python
# Зрозумій дані спочатку!
# - Які топ товари?
# - Розподіл розмірів транзакцій?
# - Temporal patterns?
# - Customer segments?
```

### 2. Налаштуй параметри під бізнес

```python
# min_support: залежить від мети
# - Cross-selling: 1-5% (rare but valuable)
# - Store layout: 10%+ (affect many customers)

# min_confidence: 
# - Aggressive campaigns: 50%+
# - Conservative: 70%+

# min_lift:
# - Must have: > 1.2
```

### 3. Фільтруй тривіальні правила

```python
# Видалити очевидні асоціації
trivial_pairs = [
    ('Coffee', 'Sugar'),
    ('Bread', 'Butter'),
    ('Chips', 'Salsa'),
    # ... domain knowledge
]

def is_trivial(ant, cons):
    for a, c in trivial_pairs:
        if a in ant and c in cons:
            return True
    return False

rules_filtered = rules[
    ~rules.apply(lambda r: is_trivial(r['antecedents'], r['consequents']), axis=1)
]
```

### 4. Segment-specific analysis

```python
# Різні сегменти → різні паттерни
# Champions vs Budget Shoppers
# Weekday vs Weekend
# Morning vs Evening
```

### 5. Temporal analysis

```python
# Паттерни змінюються з часом!
# Seasonal products
# Trending items
# Holiday effects

# Rolling window analysis
for month in ['Jan', 'Feb', 'Mar']:
    monthly_data = filter_by_month(df, month)
    rules_monthly = run_mba(monthly_data)
    # Compare changes
```

### 6. Category-level analysis

```python
# Не тільки products, але й categories
# "Dairy → Bakery"
# Допомагає в macro store layout
```

### 7. Validate з A/B testing

```python
# Не всі правила працюють на практиці!
# Test перед повним rollout
```

### 8. Monitor continuously

```python
# Dashboard з key metrics
# - Top rules
# - Changes over time
# - ROI tracking
```

### 9. Combine з іншими даними

```python
# MBA + Weather data
# MBA + Events (holidays, sports)
# MBA + Customer demographics
```

### 10. Communicate results

```python
# Business stakeholders не знають lift/support
# Translate в бізнес-мову:
# "75% людей що купують X також купують Y"
# "Потенційна додаткова виручка $50K/місяць"
```

---

## Кейси з практики

### Case 1: Amazon "Frequently Bought Together"

**Проблема:** Збільшити average order value.

**Рішення:**
- MBA на purchase history
- Real-time recommendations
- "Add both to cart" button

**Результат:**
- +35% додаткових продажів
- +20% average basket size

### Case 2: Walmart Store Layout

**Проблема:** Оптимізувати розташування товарів.

**Рішення:**
- MBA на POS data
- Розмістити асоційовані товари разом
- "End cap" displays для bundles

**Результат:**
- +15% sales для paired products
- Покращений customer flow

### Case 3: Starbucks Food Pairing

**Проблема:** Збільшити food sales.

**Рішення:**
- MBA: Coffee types → Food items
- "Pairs well with" recommendations
- Training baristas на upsell

**Результат:**
- +25% food attachment rate
- Higher customer satisfaction

---

## Поширені помилки ❌

### 1. Ігнорувати domain knowledge

```python
# ❌ Приймати всі правила literally
# Деякі очевидні, деякі nonsensical

# ✅ Filter через business logic
```

### 2. Не тестувати рекомендації

```python
# ❌ Implement без validation
# Може не працювати в реальності

# ✅ A/B test спочатку
```

### 3. Занадто низький min_support

```python
# ❌ min_support=0.001
# Мільйони слабких правил

# ✅ Розумний поріг на основі бізнесу
```

### 4. Забути про causality

```python
# ❌ "A causes B"
# Correlation ≠ Causation!

# ✅ "A and B часто разом" (може бути 3rd factor)
```

### 5. Static analysis

```python
# ❌ Одноразовий аналіз
# Паттерни змінюються!

# ✅ Regular updates (monthly/quarterly)
```

---

## Інструменти та технології

### Python Libraries
- **mlxtend** — Apriori, FP-Growth
- **pandas** — Data manipulation
- **networkx** — Graph visualization
- **matplotlib/seaborn** — Plotting

### Business Intelligence
- **Tableau** — Dashboards
- **Power BI** — Reporting
- **Looker** — Analytics

### Big Data
- **Spark MLlib** — Distributed MBA
- **Apache Mahout** — Scalable algorithms

---

## Пов'язані теми

- [[01_Apriori]] — основний алгоритм
- [[02_FP-Growth]] — швидша альтернатива
- [[Recommendation_Systems]] — персоналізовані рекомендації
- [[Customer_Segmentation]] — RFM analysis
- [[A_B_Testing]] — валідація рекомендацій

## Ресурси

- [mlxtend Market Basket Analysis](http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/)
- [Market Basket Analysis Tutorial](https://www.kaggle.com/code/datatheque/market-basket-analysis-tutorial)
- [Practical Guide to Market Basket Analysis](https://towardsdatascience.com/market-basket-analysis-978ac064d8c6)

---

## Ключові висновки

> Market Basket Analysis — практичне застосування association rules mining для виявлення паттернів покупок та оптимізації retail/e-commerce стратегій через product placement, cross-selling, bundling та personalized marketing.

**Бізнес-цінність:**
- 🛒 Оптимізація store layout
- 💰 Збільшення sales через cross-selling
- 🎯 Targeted marketing campaigns
- 📦 Smart bundling strategies
- 📊 Inventory optimization

**Процес:**
1. **Збір даних** — POS, e-commerce logs
2. **Preprocessing** — очистити, transform
3. **EDA** — зрозуміти паттерни
4. **Mining** — Apriori/FP-Growth
5. **Filtering** — strong + non-trivial rules
6. **Interpretation** — бізнес insights
7. **Recommendations** — actionable advice
8. **Validation** — A/B testing
9. **Implementation** — rollout
10. **Monitor** — track impact

**Ключові метрики:**
- **Support:** як часто разом
- **Confidence:** ймовірність купівлі
- **Lift:** сила зв'язку (>1 = позитивна кореляція)

**Застосування:**
- Product placement в магазині
- "Frequently bought together" recommendations
- Bundle пропозиції
- Email campaigns (персоналізовані)
- Inventory management

**Best Practices:**
- Segment-specific analysis (RFM)
- Temporal patterns (seasons, trends)
- Domain knowledge filtering
- A/B testing validation
- ROI calculation
- Continuous monitoring

**Найважливіше:**
- Translate technical metrics → business value
- Test перед implementation
- Combine з іншими даними
- Monitor continuously
- Update regularly

---

#ml #unsupervised-learning #market-basket-analysis #association-rules #retail #e-commerce #cross-selling #recommendations #business-analytics
