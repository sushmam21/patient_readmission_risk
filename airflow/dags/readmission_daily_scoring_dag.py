from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "ml",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    dag_id="readmission_daily_scoring",
    default_args=default_args,
    start_date=datetime(2025, 8, 1),
    schedule_interval="@daily",
    catchup=False,
) as dag:

    # Example: re-train or refresh pipeline daily (replace with real scoring job)
    retrain = BashOperator(
        task_id="retrain_model",
        bash_command="python /opt/airflow/predictive_readmission/src/train.py"
    )

    retrain
