# Google Cloud ML Engine example

This repository provides step-by-step instructions
for training a
TensorFlow model in Google Cloud Platform's Machine Learning Engine.
We make use of sample code from [cloudml-samples repository](https://github.com/GoogleCloudPlatform/cloudml-samples).

We will train a binary classifier to predict successful marketing
phone calls of a Portuguese banking institution. The dataset is available
at [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing).

## Prerequisites

* a GCP project with billing enabled

## Instructions

### Get the data

1. Navigate to your [Google Cloud Console](https://console.cloud.google.com) and select your project

* Open the Cloud Shell and clone the GitHub repository
```
git clone https://github.com/mikkokemppainen/gcp-mlengine-example.git
```

* Create a temporary folder to hold the data. In this folder, run
```
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip
```

* Unzip
```
unzip bank-additional.zip
```

* Set the environment variables
```
BUCKET_NAME=$GOOGLE_CLOUD_PROJECT-mlengine
REGION=europe-west1
```

* Create a storage bucket to
```
gsutil mb -l $REGION gs://$BUCKET_NAME
```

* Copy the data files
```
gsutil cp bank-additional/bank-additional-full.csv gs://$BUCKET_NAME/data/
```

* Delete the temporary folder and its contents

### Data preparation

In this section we split the data into training and evaluation sets
using Cloud Datalab.

1. Enable Compute Engine API
```
gcloud services enable compute.googleapis.com
```

2. Create a Datalab instance for data preparation
```
datalab create --zone europe-west1-b --disk-size-gb 20 --no-create-repository my-datalab
```

3. Open the Datalab web UI and upload the `data_preparation.ipynb` notebook in the notebooks folder

or, alternatively, create a new notebook and follow the instructions below

```
import os
project_id = os.environ['VM_PROJECT']
```

* In the notebook, read the data file from storage
```
%gcs read -o gs://$GOOGLE_CLOUD_PROJECT-mlengine/data/bank-additional-full.csv -v data_file
```

* Import required libraries
```
import pandas as pd
from io import BytesIO
```

* Read the data into a pandas dataframe
```
data = pd.read_csv(BytesIO(data_file), sep=';')
```

* Drop the `duration` column (see the explanation at the dataset source)
```
data.drop(columns='duration', inplace=True)
```

* Split the data into training and evaluation sets
```
data_train = data.sample(frac=0.7)
data_eval = data.drop(data_train.index)
```

* Write into CSV
```
data_train.to_csv('bank_data_train.csv', index=False, header=False)
data_eval.to_csv('bank_data_eval.csv', index=False, header=False)
```

4. Copy training and evaluation datasets into Cloud Storage
```
!gsutil cp bank_data*.csv gs://$GOOGLE_CLOUD_PROJECT-mlengine/data/
```

* Close the Datalab web UI, interrupt the server of Cloud Shell with `CTRL-C` and delete the Datalab instance by
```
datalab delete --delete-disk my-datalab
```

### Building the model

Our model is based on the template `model.py`

1. In the beginning of  we list the columns from our data
```py
CSV_COLUMNS = [
    'age', 'job', 'marital', 'education', 'default',
    'housing', 'loan', 'contact', 'month', 'day_of_week',
    'campaign', 'pdays', 'previous', 'poutcome', 'emp_var_rate',
    'cons_price_idx', 'cons_conf_idx', 'euribor3m', 'nr_employed', 'subscribed'
]
```

2. set the default values
```py
CSV_COLUMN_DEFAULTS = [[0], [''], [''], [''], [''],
                       [''], [''], [''], [''], [''],
                       [0], [0], [0], [''], [0.0],
                       [0.0], [0.0], [0.0], [0.0], ['']]
```

3. The target (or label) column is
```py
LABEL_COLUMN = 'subscribed'
LABELS = ['yes', 'no']
```

4. in `model.py`
```py
INPUT_COLUMNS = [
    tf.feature_column.numeric_column('age'),
    tf.feature_column.categorical_column_with_vocabulary_list(
        'job', ['admin.', 'blue-collar', 'entrepreneur', 'housemaid',
                'management', 'retired', 'self-employed', 'services',
                'student', 'technician', 'unemployed', 'unknown']),
    tf.feature_column.categorical_column_with_vocabulary_list(
        'marital', ['divorced', 'married', 'single', 'unknown']),
    tf.feature_column.categorical_column_with_vocabulary_list(
        'education', ['basic.4y', 'basic.6y', 'basic.9y',
                      'high.school', 'illiterate', 'professional.course',
                      'university.degree', 'unknown']),
    tf.feature_column.categorical_column_with_vocabulary_list(
        'default', ['yes', 'no']),
    tf.feature_column.categorical_column_with_vocabulary_list(
        'housing', ['yes', 'no']),
    tf.feature_column.categorical_column_with_vocabulary_list(
        'loan', ['yes', 'no']),
    tf.feature_column.categorical_column_with_vocabulary_list(
        'contact', ['cellular', 'telephone']),
    tf.feature_column.categorical_column_with_vocabulary_list(
        'month', ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                    'jul', 'aug', 'sep', 'oct', 'nov', 'dec']),
    tf.feature_column.categorical_column_with_vocabulary_list(
        'day_of_week', ['mon', 'tue', 'wed', 'thu', 'fri']),
    tf.feature_column.numeric_column('campaign'),
    tf.feature_column.numeric_column('pdays'),
    tf.feature_column.numeric_column('previous'),
    tf.feature_column.categorical_column_with_vocabulary_list(
        'poutcome', ['failure', 'nonexistent', 'success']),
    tf.feature_column.numeric_column('emp_var_rate'),
    tf.feature_column.numeric_column('cons_price_idx'),
    tf.feature_column.numeric_column('cons_conf_idx'),
    tf.feature_column.numeric_column('euribor3m'),
    tf.feature_column.numeric_column('nr_employed'),
]
```


### Train the model

In this section we train the TensorFlow model on Cloud ML Engine.

1. Set the environment variables for data
```
TRAIN_DATA=gs://$BUCKET_NAME/data/bank_data_train.csv
EVAL_DATA=gs://$BUCKET_NAME/data/bank_data_eval.csv
```

* In the repository folder, set the environment variables for training job
```
HPTUNING_CONFIG=$(pwd)/hptuning_config.yaml
JOB_NAME=bank_marketing_hptune_1
OUTPUT_PATH=gs://$BUCKET_NAME/$JOB_NAME
```

4. Run training with hyperparameter tuning
```
gcloud ml-engine jobs submit training $JOB_NAME \
    --stream-logs \
    --job-dir $OUTPUT_PATH \
    --runtime-version 1.8 \
    --config $HPTUNING_CONFIG \
    --module-name trainer.task \
    --package-path trainer/ \
    --region $REGION \
    --scale-tier STANDARD_1 \
    -- \
    --train-files $TRAIN_DATA \
    --eval-files $EVAL_DATA \
    --train-steps 10000 \
    --eval-steps 1000 \
    --verbosity DEBUG
```

### Deploy the model

1. Set the environment variable
```
MODEL_NAME=bank_marketing
```

2. Create a model in ML Engine
```
gcloud ml-engine models create $MODEL_NAME --regions=$REGION
```

3. Select the job output to use and look up the path to model binaries
```
gsutil ls -r $OUTPUT_PATH/export
```

4. Set the environment variable with the correct value for `<timestamp>`
```
MODEL_BINARIES=$OUTPUT_PATH/export/bank_marketing/<timestamp>/
```

5. Create a version of the model
```
gcloud ml-engine versions create v1 \
    --model $MODEL_NAME \
    --origin $MODEL_BINARIES \
    --runtime-version 1.8
```

6.  In the repository folder, inspect the test instance
```
cat test.json
```

7. Get the prediction for the test instance
```
gcloud ml-engine predict \
    --model $MODEL_NAME \
    --version v1 \
    --json-instances \
    test.json
```
