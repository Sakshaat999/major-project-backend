import pandas as pd# type: ignore
from sklearn.model_selection import train_test_split# type: ignore
from sklearn.naive_bayes import GaussianNB# type: ignore
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,precision_score,recall_score,f1_score# type: ignore
from sklearn.preprocessing import LabelEncoder# type: ignore
from sklearn.impute import SimpleImputer# type: ignore

df = pd.read_csv('HAM10000_metadata.csv')
features_cols = df.columns.tolist()
target_label = 'dx_type'
if target_label in features_cols:
    features_cols.remove(target_label)
features = df[features_cols].copy()
for col in features.columns:
    if features[col].dtype == 'object':
        le = LabelEncoder()
        features[col] = le.fit_transform(features[col])
imputer = SimpleImputer(strategy='mean')
features_imputed = pd.DataFrame(imputer.fit_transform(features), columns=features.columns)
X_train, X_test, y_train, y_test = train_test_split(features_imputed, df[target_label], test_size=0.2, random_state=42)

naive_bayes_classifier = GaussianNB()

naive_bayes_classifier.fit(X_train, y_train)

y_pred = naive_bayes_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f'\nAccuracy: {accuracy:.2f}')
print('\nConfusion Matrix:')
print(conf_matrix)
print('\nClassification Report:')
print(classification_rep)


precision_optimal = precision_score(y_test, y_pred, average='weighted')
recall_optimal = recall_score(y_test, y_pred, average='weighted')
f1_optimal = f1_score(y_test, y_pred, average='weighted')

print(f'\nWeighted Precision for Optimal k: {precision_optimal:.2f}')
print(f'Weighted Recall for Optimal k: {recall_optimal:.2f}')
print(f'Weighted F1 Score for Optimal k: {f1_optimal:.2f}')