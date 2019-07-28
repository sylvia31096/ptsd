# PTSD Classfier backend

Version : 0.1

The classifier backend leverages the open source tool MLFlow. 

## Files and directory structure

*./data/transcripts/PTSD_data.csv* : The initial dataset annotated. Export the relevant spreadsheet from csv.

*./Dockerfile* : Initial Dockerfile of the solution.

*./multi_label_lg_data_pipeline.ipynb* : Multilabel reference pipeline, very simple. 

*./MLProject* : MLflow reference project.

*./train.py* : The training source code file.

## How to run it

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