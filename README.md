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

* Create a storage bucket to hold the data
```
gsutil mb -l $REGION gs://$BUCKET_NAME
```

* Copy the data files
```
gsutil cp bank-additional/bank-additional-full.csv gs://$BUCKET_NAME/data/
```

* Delete the temporary folder and its contents

### Data preparation

In this section we split the data into training, evaluation, and testing sets using Cloud Datalab.

1. Enable Compute Engine API
```
gcloud services enable compute.googleapis.com
```

2. Create a Datalab instance for data preparation
```
datalab create --zone europe-west1-b --disk-size-gb 20 --no-create-repository my-datalab
```

3. Open the Datalab web UI and upload the `data_preparation.ipynb` notebook in the notebooks folder

4. Run the notebook


### Building the model

Our model is based on the template `model_template.py`

1. In the beginning of `model.py` we list the columns from our data
```py
CSV_COLUMNS = [
    'age', 'job', 'marital', 'education', 'default',
    'housing', 'loan', 'contact', 'month', 'day_of_week',
    'campaign', 'pdays', 'previous', 'poutcome', 'emp_var_rate',
    'cons_price_idx', 'cons_conf_idx', 'euribor3m', 'nr_employed', 'subscribed'
]
```

* Then we set the default values paying attention to data types
```py
CSV_COLUMN_DEFAULTS = [[0], [''], [''], [''], [''],
                       [''], [''], [''], [''], [''],
                       [0.0], [0.0], [0.0], [''], [0.0],
                       [0.0], [0.0], [0.0], [0.0], ['']]
```

* The target (or label) column is
```py
LABEL_COLUMN = 'subscribed'
LABELS = ['no', 'yes']
```

* We then list the input columns with their correct interpretations
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

* In `build_estimator` function we define how the input columns are transformed into actual feature columns
```py
# Continuous columns can be converted to categorical via bucketization
age_buckets = tf.feature_column.bucketized_column(
    age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75])
feature_columns = [
    # Use embedding columns for high dimensional vocabularies
    tf.feature_column.embedding_column(age_buckets, dimension=embedding_size),
    tf.feature_column.embedding_column(job, dimension=embedding_size),
    # Use indicator columns for low dimensional vocabularies
    tf.feature_column.indicator_column(marital),
    tf.feature_column.embedding_column(education, dimension=embedding_size),
    tf.feature_column.indicator_column(default),
    tf.feature_column.indicator_column(housing),
    tf.feature_column.indicator_column(loan),
    tf.feature_column.indicator_column(contact),
    tf.feature_column.embedding_column(month, dimension=embedding_size),
    tf.feature_column.indicator_column(day_of_week),
    campaign,
    pdays,
    previous,
    tf.feature_column.indicator_column(poutcome),
    emp_var_rate,
    cons_price_idx,
    cons_conf_idx,
    euribor3m,
    nr_employed
]
```

* The `build_estimator` function returns an estimator with our choice of settings, such as optimizer and batch normalization
```py
return tf.estimator.DNNClassifier(
     config=config,
     feature_columns=feature_columns,
     optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
     batch_norm=True,
     hidden_units=hidden_units or [32, 24, 16, 8])
```


### Defining the training job

The various training parameters are defined in `task.py`.
Their values are fed to the training job via hyperparameter
tuning, which are specified in `hptuning_config.yaml`.
Because of the imbalanced target distribution in makes sense
to aim to maximize not the accuracy, but the area under
precision-recall curve.


### Train the model

In this section we train the TensorFlow model on Cloud ML Engine.

1. Enable the Cloud ML Engine API
```
gcloud services enable ml.googleapis.com
```

* Set the environment variables for data
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
    --runtime-version 1.10 \
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
MODEL_NAME=bank-marketing
```

2. Create a model in ML Engine
```
gcloud ml-engine models create $MODEL_NAME --regions=$REGION
```

3. Select the job output to use and look up the path to model binaries
```
gsutil ls -r $OUTPUT_PATH/
```

4. Set the environment variable with the correct value for `trial_number` and `<timestamp>`
```
MODEL_BINARIES=$OUTPUT_PATH/<trial_number>/export/bank-marketing/<timestamp>/
```

5. Create a version of the model
```
gcloud ml-engine versions create v1 \
    --model $MODEL_NAME \
    --origin $MODEL_BINARIES \
    --runtime-version 1.10
```


### Predict

1. Upload and open the `model_predictions.ipynb` notebook in Datalab

2. Run it and inspect the results

### Clean up

1. Close the Datalab web UI, interrupt the server of Cloud Shell with `ctrl+c` and delete the Datalab instance by
```
datalab delete --delete-disk my-datalab
```
