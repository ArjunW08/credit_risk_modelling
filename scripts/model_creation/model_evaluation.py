from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

def train_xgboost_classifier(X_train, y_train):
    xgb = XGBClassifier(
        objective='multi:softmax', 
        num_class=4, 
        random_state=42,
        n_jobs=-1)

    param_distributions = {
        'n_estimators': randint(50, 400),          # Total number of trees
        'learning_rate': uniform(0.01, 0.29),       # Step size shrinkage (0.01 to 0.30)
        'max_depth': randint(3, 22),               # Maximum tree depth
        'subsample': uniform(0.6, 0.4),            # Row sampling ratio (0.6 to 1.0)
        'colsample_bytree': uniform(0.6, 0.4),     # Column sampling ratio per tree (0.6 to 1.0)
        'gamma': uniform(0, 5)                     # Minimum loss reduction to split
    }

    scorer = make_scorer(f1_score, average='macro')

    random_search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_distributions,
        scoring=scorer,
        cv=5,
        n_jobs=-1,
        verbose=2
    )

    random_search.fit(X_train, y_train)

    return random_search


def evaluate_classifier(model, X_test, y_test, model_name):
    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)

    print(f"\n{model_name} Performance")
    print(f"Accuracy: {accuracy:.2f}")
    print(report)