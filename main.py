from flask import Flask, request
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score
app = Flask(__name__)



@app.route('/grid_search', methods=['POST'])
def grid_search():

    f1 = make_scorer(f1_score, average="weighted")

    # parameter possible
    params = {
        "n_estimators": [25, 30, 35],
        "max_depth": [15, 20, 25],
        "min_samples_leaf": [3, 4, 5]
    }

    # Obtenir les données de la requête
    data = request.json

    # utiliser grid search pour obtenir les meilleur paramètre avec Random forest
    grid_search = GridSearchCV(RandomForestClassifier(), param_grid=params, scoring=f1, cv=5)
    grid_search.fit(data["train features"], data["train labels"])


    return grid_search.best_params_


if __name__ == '__main__':
    app.run()
