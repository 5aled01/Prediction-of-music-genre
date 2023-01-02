from flask import Flask, request
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score
from zeep import Client

app = Flask(__name__)

# Load the WSDL file
with open('gridsearch.wsdl', 'r') as f:
    wsdl = f.read()
client = Client(wsdl=wsdl)

# Get the service interface
service = client.service

@app.route('/gridsearch', methods=['POST'])
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

    # Convert the result to the format required by the WSDL file
    result = {
        'n_estimators': grid_search.best_params_['n_estimators'],
        'max_depth': grid_search.best_params_['max_depth'],
        'min_samples_leaf': grid_search.best_params_['min_samples_leaf']
    }

    # Call the service with the data and the result
    service.GridSearchOperation(data, result)

    return result

if __name__ == '__main__':
    app.run()
