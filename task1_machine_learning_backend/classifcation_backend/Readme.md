# PTSD Classfier backend

Version : 0.1

The classifier backend leverages the open source tool MLFlow. 
 
Please get familiarized to easily understand this project. MLFlow website available in: https://mlflow.org/

## Load Data
Add under the folder ./data/transcripts with the filename PTSD_data.csv the annotated standard dataset.
 

## Files and directory structure

*./data/transcripts/PTSD_data.csv* : The initial dataset annotated. Export the relevant spreadsheet from csv.

*./Dockerfile* : Initial Dockerfile of the solution - Not functional.

*./conda.yaml* : Conda file w/ all the project dependencies.

*./multi_label_lg_data_pipeline.ipynb* : Multilabel reference pipeline, with the step by step implementation. 

*./MLProject* : MLflow reference project.

*./train.py* : The training source code file.

## How to run training

`$ mlflow run .`

## How to run a listening prediction api
Using the id of the model and the model name you can run the following commnad :

`$  mlflow models serve -m runs:/bd925a4361fc4910af3f318b0293b1bc/modelA1/ -p 1234  .`


## How to run a prediction

`curl http://127.0.0.1:1234/invocations -H 'Content-Type: application/json' -d '{
    "columns": ["text"],
    "data": [["hey"]]
}'`

### Download the annotation data spreadsheet and export it as it is. 


### To run training command you can use the command below from within the directory of the project:
 
 ` 
╰─$ python ./train.py                    

... Processing A1

Test accuracy is 0.9375

... Processing B1
Test accuracy is 1.0

... Processing B2
Test accuracy is 1.0
...

... Processing G1
Test accuracy is 1.0
` 