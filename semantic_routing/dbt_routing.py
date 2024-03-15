"""
https://github.com/aurelio-labs/semantic-router/blob/main/docs/06-threshold-optimization.ipynb
https://github.com/aurelio-labs/semantic-router/blob/main/docs/03-basic-langchain-agent.ipynb
"""
import json

from semantic_router import Route, RouteLayer

from semantic_routing.ollama_encoder import OllamaEncoder

TRAINED_THRESHOLDS = {
    'dbt_metadata': 0.686868686868687, 'dbt_code': 0.7164983164983165,
    'dbt_columns': 0.43434343434343436, 'dbt_descriptions': 0.5656565656565657,
    'dbt_upstream_models': 0.6363636363636365, 'dbt_downstream_models': 0.7124987246199368
}

dbt_metadata = Route(
    name="metadata",
    score_threshold=TRAINED_THRESHOLDS['dbt_metadata'],
    utterances=[
        "What type of dbt model is this?",
        "What schema does this dbt model belong to?",
        "How many bytes does this dbt model contain?",
        "How many rows does this dbt model have?",
        "When was this dbt model created?",
        "What is the unique id of this dbt model?",
    ],
)

dbt_code = Route(
    name="code",
    score_threshold=TRAINED_THRESHOLDS['dbt_code'],
    utterances=[
        "How were tables joined to generate this DBT model?",
        "What post_hooks does this dbt model have?",
        "Explain the logic behind the DBT model.",
        "How are rows filtered in the DBT model?",
        "Is this DBT model an incremental model?"
        "What source tables were used to generate this DBT model?",
    ],
)

dbt_columns = Route(
    name="columns",
    score_threshold=TRAINED_THRESHOLDS['dbt_columns'],
    utterances=[
        "How many columns does this dbt model have?",
        "What is the description for this column on this dbt model?",
        "What is the data type for this column on this dbt model?",
        "Does this DBT model have a column like this?",
        "Does this DBT model have any date columns?",
        "What is the primary key of ths DBT model?",
    ],
)

dbt_descriptions = Route(
    name="descriptions",
    score_threshold=TRAINED_THRESHOLDS['dbt_descriptions'],
    utterances=[
        "What is the spark configuration for this dbt model?",
        "What is the location of this dbt model?",
        "What Serde library is used for this dbt model?",
        "What is the input format for this dbt model?",
        "What is the output format for this dbt model?",
        "What are the statistics for this dbt model?"
    ],
)

dbt_upstream_models = Route(
    name="upstream_models",
    score_threshold=TRAINED_THRESHOLDS['dbt_upstream_models'],
    utterances=[
        "What upstream models does this dbt model have?",
        "Does this dbt model have any upstream models?",
        "Is this dbt model upstream to that dbt model?",
        "How many upstream models does this dbt model have?",
        "Check the upstream model flow for this dbt model and that dbt model. Will that dbt model be impacted if this dbt model changes?",
        "Does this dbt model have any upstream models named like this?"
    ],
)

dbt_downstream_models = Route(
    name="downstream_models",
    score_threshold=TRAINED_THRESHOLDS['dbt_downstream_models'],
    utterances=[
        "What downstream models does this dbt model have?",
        "Does this dbt model have any downstream models?",
        "Is this dbt model downstream to that dbt model?",
        "How many downstream models does this dbt model have?",
        "Check the downstream model flow for this dbt model from that dbt model. Will this dbt model be impacted if that dbt model changes?",
        "Does this dbt model have any downstream models named like this?"
    ],
)

routes = [dbt_metadata, dbt_code, dbt_columns, dbt_descriptions, dbt_upstream_models, dbt_downstream_models]
utterances = [
    {
        route.name: [route.utterances]
    }
    for route in routes
]


def get_route_layer():
    # return RouteLayer(encoder=OpenAIEncoder(), routes=routes)
    return RouteLayer(encoder=OllamaEncoder(), routes=routes)


if __name__ == '__main__':
    # unpack the test data
    with open('./semantic_routing/route_training.json') as f:
        train_data = json.load(f)
    with open('./semantic_routing/route_validation.json') as f:
        test_data = json.load(f)
    X_train, y_train = zip(*[(route['question'], route['route']) for route in train_data])
    X_test, y_test = zip(*[(route['question'], route['route']) for route in test_data])

    # Evaluate using the default thresholds
    rl = get_route_layer()
    route_thresholds = rl.get_thresholds()
    print("Default route thresholds:", route_thresholds)

    accuracy = rl.evaluate(X=X_train, y=y_train)
    print(f"Default Train Accuracy: {accuracy * 100:.2f}%")
    accuracy = rl.evaluate(X=X_test, y=y_test)
    print(f"Default Test Accuracy: {accuracy * 100:.2f}%")

    # Call the fit method
    rl.fit(X=X_train, y=y_train)
    route_thresholds = rl.get_thresholds()
    print("Updated route thresholds:", route_thresholds)

    accuracy = rl.evaluate(X=X_train, y=y_train)
    print(f"Fit Train Accuracy: {accuracy * 100:.2f}%")
    accuracy = rl.evaluate(X=X_test, y=y_test)
    print(f"Fit Test Accuracy: {accuracy * 100:.2f}%")
