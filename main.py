import os
from datetime import datetime
from pickle import dumps

import boto3
from feast import FeatureStore
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres import (
    PostgreSQLOfflineStore,
)
from train import train


def retrieveTrainingDataset(startDate, endDate, deploymentName, store, ioNames):
    fv = store.get_feature_view(deploymentName)
    source = fv.batch_source
    dataset = PostgreSQLOfflineStore.pull_all_from_table_or_query(
        config=store.config,
        data_source=source,
        join_key_columns=fv.join_keys,
        timestamp_field=source.timestamp_field,
        feature_name_columns=ioNames,
        start_date=startDate,
        end_date=endDate,
    ).to_df()
    return dataset.drop(labels=["id", "timestamp"], axis="columns")


def store_model(model):
    serialized = dumps(model)
    s3 = boto3.Session().resource(
        service_name="s3",
        endpoint_url=os.environ["FEAST_S3_ENDPOINT_URL"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )
    s3Object = s3.Object(
        "models",
        "callcenter-linear-"
        + datetime.now().isoformat(timespec="minutes")
        + "/model.joblib",
    )
    s3Object.put(Body=serialized)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    startDate = datetime.fromisoformat(os.environ["startDate"].replace("Z", "+00:00"))
    endDate = datetime.fromisoformat(os.environ["endDate"].replace("Z", "+00:00"))
    deploymentName = os.environ["deploymentName"]
    ioNames = os.environ["ioNames"].split(",")
    store = FeatureStore(repo_path=".")
    dataset = retrieveTrainingDataset(
        startDate, endDate, deploymentName, store, ioNames
    )
    model = train(dataset)
    store_model(model)
