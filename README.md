# CV parser

This project is implementing customized Name Entity Recognition (NER) by spaCy. Recreate 11 classes of name entities and train the model.

## tech stack
- Amazon S3
- Dataturks
- Heroku
- python flask 
- spaCy

## customized name entities description
```
Name: Name of user
Contact: Contact of user, like phone number, email address
Location: Location where user based in or where user want to relocate
Title: Job title user has ever had
Company: Company user has ever worked at
TimePeriod: Period of the time that can be found in CV
Degree: Education
College: Education
Skills: Technical stacks, soft-skills
Years: Preset a year long, like "3+ years"
Others: Links like LinkedIn profile or others
```

## Train your own model

If you want to train your own model, I also wrote a command line tool to train model easily

### Installation
```
python version: 3.7.4
```

```
git clone https://github.com/samuelTyh/CVparser.git
cd CVparser
pip install -r requirements.txt
```

### Configuration setting
Set testing data size, iteration times, and early stopping times in `config.py`
```
# default hyperparameter in config.py

test_size = 0.3
n_iter = 300
early_stopping = 50
```

### Train
```
# predict without training
$ python app_cmd.py -f file_you_want_to_parse.pdf -M lib/model_choose_optional/

# train model first then predict
$ python app_cmd.py -f file_you_want_to_parse.pdf
```


## demo site
https://cvparser-demo.herokuapp.com/

```
Press Choose file and upload buttons to upload CV (only support English version)
Wait for the result from model prediction
```
