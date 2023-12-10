from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


# Załadujmy przykładowy zbiór danych (iris)
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Podzielmy zbiór na dane treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Różne wartości k i miary odległości
k_values = [3, 5, 7]
distance_metrics = ['euclidean', 'manhattan', 'minkowski']

# Testujemy dla różnych wartości k i miar odległości
for k in k_values:
    for metric in distance_metrics:
        # Utwórz klasyfikator kNN z danymi parametrami
        knn_classifier = KNeighborsClassifier(n_neighbors=k, metric=metric)

        # Dopasuj klasyfikator do danych treningowych
        knn_classifier.fit(X_train, y_train)

        # Przewiduj klasy dla danych testowych
        y_pred = knn_classifier.predict(X_test)

        # Oceniaj dokładność klasyfikatora
        accuracy = accuracy_score(y_test, y_pred)
        print(f'k={k}, Metryka={metric}, Dokładność kNN: {accuracy}')




# Załadujmy przykładowy zbiór danych (Wine)
wine = datasets.load_wine()
X = wine.data
y = wine.target

# Podzielmy zbiór na dane treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Różne wartości k i miary odległości
k_values = [3, 5, 7]
distance_metrics = ['euclidean', 'manhattan', 'minkowski']

# Testujemy dla różnych wartości k i miar odległości
for k in k_values:
    for metric in distance_metrics:
        # Utwórz klasyfikator kNN z danymi parametrami
        knn_classifier = KNeighborsClassifier(n_neighbors=k, metric=metric)

        # Dopasuj klasyfikator do danych treningowych
        knn_classifier.fit(X_train, y_train)

        # Przewiduj klasy dla danych testowych
        y_pred = knn_classifier.predict(X_test)

        # Oceniaj dokładność klasyfikatora
        accuracy = accuracy_score(y_test, y_pred)
        print(f'k={k}, Metryka={metric}, Dokładność kNN: {accuracy}')



# Definiuj zbiory danych
datasets = [(iris.data, iris.target, 'Iris'), (wine.data, wine.target, 'Wine')]

# Parametry do przeszukania
param_grid = {'n_neighbors': [3, 5, 7],
              'metric': ['euclidean', 'manhattan', 'minkowski']}

# Testujemy dla różnych zbiorów danych
for X, y, dataset_name in datasets:
    # Utwórz klasyfikator kNN
    knn_classifier = KNeighborsClassifier()

    # Utwórz obiekt GridSearchCV
    grid_search = GridSearchCV(knn_classifier, param_grid, cv=5, scoring='accuracy')

    # Przeszukaj przestrzeń parametrów
    grid_search.fit(X, y)

    # Wydrukuj najlepsze parametry
    print(f'Najlepsze parametry dla zbioru danych {dataset_name}: {grid_search.best_params_}')
    print(f'Najlepsza dokładność dla zbioru danych {dataset_name}: {grid_search.best_score_}\n')