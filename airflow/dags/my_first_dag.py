from airflow import DAG
from airflow.operators.empty import EmptyOperator  # Use EmptyOperator for Airflow 2.x
from datetime import datetime

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 1, 21),
    'retries': 1,
}

# Define the DAG
dag = DAG(
    'my_first_dag',
    default_args=default_args,
    description='A simple first DAG',
    schedule_interval='@daily',
)

# Define tasks
start_task = EmptyOperator(task_id='start', dag=dag)
end_task = EmptyOperator(task_id='end', dag=dag)

# Set task dependencies
start_task >> end_task
