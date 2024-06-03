import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_fscore_support, plot_confusion_matrix, accuracy_score
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# Завантаження даних
heart_data = pd.read_csv('heart.csv')

# Підготовка даних
encoder = OneHotEncoder(sparse=False)
X_categorical = encoder.fit_transform(heart_data[['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']])
categorical_features = encoder.get_feature_names_out()
X_numerical_original = heart_data[['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']].values
numerical_features = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
X_original_combined = np.hstack([X_categorical, X_numerical_original])
y = heart_data['HeartDisease'].values

# Розділення на тренувальну та тестову вибірки
X_train_orig, X_test_orig, y_train, y_test = train_test_split(X_original_combined, y, test_size=0.25, random_state=42)

# Ініціалізація моделей
models = {
    "Logistic Regression": LogisticRegression(max_iter=1500),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(random_state=42),
    "Neural Network": MLPClassifier(random_state=42, max_iter=2500)
}


# Функція для побудови Learning Curve
def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None, n_jobs=None,
                        train_sizes=np.linspace(.1, 1.0, 5)):
    if axes is None:
        _, axes = plt.subplots(1, 1, figsize=(5, 5))
    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='accuracy')
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    axes.grid()
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                      train_scores_mean + train_scores_std, alpha=0.1, color="r")
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                      test_scores_mean + test_scores_std, alpha=0.1, color="g")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    axes.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    axes.legend(loc="best")
    return plt


print("================================START=================================")
# Оцінка моделей
for name, model in models.items():
    model.fit(X_train_orig, y_train)
    y_pred = model.predict(X_test_orig)
    fig, ax = plt.subplots(figsize=(5, 5))
    plot_confusion_matrix(model, X_test_orig, y_test, ax=ax, cmap='Blues')
    ax.set_title(f'Confusion Matrix for {name}')
    plt.show()
    precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    print(f"{name} - Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {fscore:.2f}")
    fig, axes = plt.subplots(1, 1, figsize=(6, 4))
    plot_learning_curve(model, f'Learning Curve for {name}', X_train_orig, y_train, axes=axes, ylim=(0.7, 1.01),
                        n_jobs=4, cv=5)
    plt.show()

print("===============================SUMMARY================================")
results = {}
for name, model in models.items():
    model.fit(X_train_orig, y_train)  # Навчання моделі
    y_pred = model.predict(X_test_orig)  # Прогнозування на тестових даних
    accuracy = accuracy_score(y_test, y_pred)  # Обчислення точності
    results[name] = accuracy

    print(f"{name}: Accuracy = {accuracy:.2f}")
print("======================================================================")
best_model = max(results, key=results.get)
print(f"Most effective model: {best_model} with accuracy = {results[best_model]:.2f}")
