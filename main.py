import pandas as pd

doc = 'C:/Users/TR/Desktop/data.csv'
data = pd.read_csv(doc)

print(data.head())

#Veri seti hakkında genel bilgi
print(data.info())

#Temel istatistikler
print(data.describe())

#Gereksiz sütunları düşür (ID sütunu)
data = data.drop(['id', 'Unnamed: 32'], axis=1)

#Hedef değişkeni ve özellikleri ayır
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

#Hedef değişkeni (diagnosis) sayısal değerlere dönüştür
y = y.map({'M': 1, 'B': 0})

#Veriyi eğitim ve test setlerine ayır
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Veriyi ölçekle
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#Modeli eğit
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

#Tahmin yap
y_pred = model.predict(X_test)

#Model performansını değerlendir
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print('Classification Report:')
print(classification_report(y_test, y_pred))


from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10, 100],
    'solver': ['lbfgs', 'liblinear']
}

grid_search = GridSearchCV(LogisticRegression(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)

#En iyi parametrelerle yeniden tahmin yap
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)

print(f'Best Accuracy: {accuracy_score(y_test, y_pred_best)}')
print('Best Confusion Matrix:')
print(confusion_matrix(y_test, y_pred_best))
print('Best Classification Report:')
print(classification_report(y_test, y_pred_best))


