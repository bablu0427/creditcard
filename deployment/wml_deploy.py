import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ibm_watson_machine_learning import APIClient

from config import MODEL_PATH


def require_env(name):
    value = os.getenv(name)
    if not value:
        raise EnvironmentError(f"Missing required environment variable: {name}")
    return value


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found: {MODEL_PATH}")

    credentials = {
        "url": require_env("IBM_WML_URL"),
        "apikey": require_env("IBM_CLOUD_API_KEY"),
    }
    space_id = require_env("IBM_WML_SPACE_ID")
    client = APIClient(credentials)
    client.set.default_space(space_id)

    metadata = {
        client.repository.ModelMetaNames.NAME: "Credit Card Approval Prediction Pipeline",
        client.repository.ModelMetaNames.TYPE: "scikit-learn_1.5",
        client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: client.software_specifications.get_id_by_name(
            "runtime-24.1-py3.11"
        ),
    }

    stored_model = client.repository.store_model(str(MODEL_PATH), meta_props=metadata)
    model_id = client.repository.get_model_id(stored_model)

    deployment_props = {
        client.deployments.ConfigurationMetaNames.NAME: "credit-card-approval-online",
        client.deployments.ConfigurationMetaNames.ONLINE: {},
    }
    deployment = client.deployments.create(model_id, meta_props=deployment_props)
    deployment_id = client.deployments.get_id(deployment)

    print("Model uploaded successfully.")
    print(f"Model ID: {model_id}")
    print(f"Deployment ID: {deployment_id}")


if __name__ == "__main__":
    main()
