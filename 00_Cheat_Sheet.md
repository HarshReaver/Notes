# DMBI Exam Cheat Sheet - One Page Reference

## What to CHANGE per question (only 3 things ever change!)
1. **Filename** or generate block (column names + ranges)
2. **Target/groupby column name**
3. **Hyperparameter** (min_support / criterion / K)

---

## TOPIC 1: Association Rule Mining
```
IMPORTS:  mlxtend.frequent_patterns → apriori, association_rules
          mlxtend.preprocessing → TransactionEncoder
FLOW:     Load CSV → groupby(ID)[Item].apply(list) → TransactionEncoder → apriori() → association_rules()
CUSTOM:   transactions = [list of lists] → same from TransactionEncoder onward
CHANGE:   filename, groupby col, item col, min_support
```

## TOPIC 2: Naive Bayes
```
IMPORTS:  sklearn.naive_bayes → GaussianNB (numbers) or MultinomialNB (text)
FLOW:     Load → LabelEncoder → train_test_split → GaussianNB().fit() → predict → classification_report
TEXT:     TfidfVectorizer → MultinomialNB
CUSTOM:   np.random.seed(42), DataFrame with 3 cols + np.where target
CHANGE:   filename, target column name, GaussianNB vs MultinomialNB
```

## TOPIC 3: Decision Tree
```
IMPORTS:  sklearn.tree → DecisionTreeClassifier, plot_tree
FLOW:     Load → LabelEncoder → train_test_split → DecisionTreeClassifier(criterion='gini') → plot_tree → evaluate
UNIQUE:   plot_tree(model, filled=True, feature_names=X.columns.tolist())
CHANGE:   filename, target col, criterion (gini/entropy)
```

## TOPIC 4: K-Means Clustering
```
IMPORTS:  sklearn.cluster → KMeans | sklearn.preprocessing → StandardScaler
FLOW:     Load → StandardScaler → Elbow Method (loop K=1-10) → KMeans(n_clusters=K) → scatter plot
EVAL:     model.inertia_ + silhouette_score()
CHANGE:   filename, feature columns, K value
```

## TOPIC 5: Hierarchical Clustering
```
IMPORTS:  sklearn.cluster → AgglomerativeClustering | scipy.cluster.hierarchy → dendrogram, linkage
FLOW:     Load → StandardScaler → linkage(ward) → dendrogram → AgglomerativeClustering(n_clusters=K) → scatter
EVAL:     silhouette_score()
CHANGE:   filename, feature columns, K value
```

---

## Universal Part A (EDA) - write this in every answer:
```python
print(df.head())
print(df.shape)
print(df.isnull().sum())
df.dropna(inplace=True)
df.hist(figsize=(10,6)); plt.tight_layout(); plt.show()
```

## Universal Part C (Classification eval) - Naive Bayes & Decision Tree:
```python
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix'); plt.show()
```

## Custom Dataset Generation Template:
```python
np.random.seed(42); n = 200
df = pd.DataFrame({
    'Col1': np.random.uniform(low, high, n),    # float
    'Col2': np.random.randint(low, high, n),     # int
    'Col3': np.random.randint(low, high, n)      # int
})
# For classification: df['Target'] = np.where(condition, 'Yes', 'No')
```
