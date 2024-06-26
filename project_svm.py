import pandas as pd# type: ignore
from sklearn.model_selection import train_test_split# type: ignore
from sklearn.svm import SVC# type: ignore
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,f1_score,recall_score,precision_score# type: ignore
from sklearn.preprocessing import LabelEncoder, StandardScaler# type: ignore
from sklearn.impute import SimpleImputer# type: ignore

df = pd.read_csv('HAM10000_metadata.csv')

df = df.drop(columns=['lesion_id', 'image_id'])

le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])
df['dx'] = le.fit_transform(df['dx'])
df['dx_type'] = le.fit_transform(df['dx_type'])
df['localization'] = le.fit_transform(df['localization'])

X = df.drop(columns=['dx'])
y = df['dx']

imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
svm_model = SVC(kernel='linear')  # You can choose a different kernel if needed
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion_mat)
print("Classification Report:\n", classification_rep)

precision_optimal = precision_score(y_test, y_pred, average='weighted')
recall_optimal = recall_score(y_test, y_pred, average='weighted')
f1_optimal = f1_score(y_test, y_pred, average='weighted')

print(f'\nWeighted Precision for Optimal k: {precision_optimal:.2f}')
print(f'Weighted Recall for Optimal k: {recall_optimal:.2f}')
print(f'Weighted F1 Score for Optimal k: {f1_optimal:.2f}')