# Naive Bayes Classifier
Classifies using probability. Assumes features are independent.
- **GaussianNB** = for numbers
- **MultinomialNB** = for text
## Using CSV Dataset (numbers)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Step 1: Load
df = pd.read_csv('diabetes.csv')  # CHANGE filename
print(df.head())
df.dropna(inplace=True)

# Encode text columns to numbers (skip if all columns are already numbers)
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

df.hist(figsize=(10,6))
plt.tight_layout()
plt.show()
# Step 2: Train - CHANGE target column name
target = 'Outcome'  # CHANGE: diabetes=Outcome, heart=target, iris=species, SocialAds=Purchased
X = df.drop(target, axis=1)
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Step 3: Results
print(classification_report(y_test, y_pred))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title('Confusion Matrix')
plt.show()
## Using CSV Dataset (text - spam/sentiment)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

df = pd.read_csv('spam.csv', encoding='latin-1')[['v1','v2']]
df.columns = ['Label','Text']

# TF-IDF converts text to numbers
tfidf = TfidfVectorizer(max_features=3000, stop_words='english')
X = tfidf.fit_transform(df['Text'])
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

print(classification_report(y_test, model.predict(X_test)))
ConfusionMatrixDisplay.from_predictions(y_test, model.predict(X_test))
plt.title('Confusion Matrix')
plt.show()
## Using Custom (Generated) Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# Step 1: Generate data - CHANGE column names, ranges, and condition
np.random.seed(42)
df = pd.DataFrame({
    'StudyHours': np.random.uniform(1, 10, 200),
    'Attendance': np.random.randint(40, 100, 200),
    'AssignmentScore': np.random.randint(20, 100, 200)
})
df['Result'] = np.where((df['StudyHours']>4) & (df['Attendance']>60), 'Pass', 'Fail')

print(df.head())
df.hist(figsize=(10,6))
plt.tight_layout()
plt.show()

# Step 2: Train + Evaluate (same every time)
target = 'Result'  # CHANGE
le = LabelEncoder()
df[target] = le.fit_transform(df[target])
X = df.drop(target, axis=1)
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)

print(classification_report(y_test, model.predict(X_test)))
ConfusionMatrixDisplay.from_predictions(y_test, model.predict(X_test))
plt.title('Confusion Matrix')
plt.show()