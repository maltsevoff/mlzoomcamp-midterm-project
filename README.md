# mlzoomcamp-midterm-project

## Problem description

The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls.

In this project I solved the problem of predicting the marketing campaing results for each client using the information about bank account and information about results of previous marketing campaing.

The classification goal is to predict if the client will subscribe a term deposit.

## Variables description

**Input variables:** 

**bank client data:** 
- 1 - age (numeric)   
- 2 - job : type of job (categorical: "admin.","unknown","unemployed","management","housemaid","entrepreneur", "student","blue-collar","self-employed","retired","technician","services")   
- 3 - marital : marital status (categorical: "married","divorced","single"; note: "divorced" means divorced or widowed)   
- 4 - education (categorical: "unknown","secondary","primary","tertiary")   
- 5 - default: has credit in default? (binary: "yes","no")   
- 6 - balance: average yearly balance, in euros (numeric)   
- 7 - housing: has housing loan? (binary: "yes","no")   
- 8 - loan: has personal loan? (binary: "yes","no")  - related with the last contact of the current campaign:  
- 9 - contact: contact communication type (categorical: "unknown","telephone","cellular")  
- 10 - day: last contact day of the month (numeric)  
- 11 - month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")  
- 12 - duration: last contact duration, in seconds (numeric)  

**other attributes:**  
- 13 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)  
- 14 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric, -1 means client was not previously contacted)   
- 15 - previous: number of contacts performed before this campaign and for this client (numeric)   
- 16 - poutcome: outcome of the previous marketing campaign (categorical: "unknown","other","failure","success")  
 
**output variable (desired target):**
- 17 - y - has the client subscribed a term deposit? (binary: "yes","no")

## Virtual environment installation

You can install virtual environment using pipenv.

Check that you have installed pipenv on your device using this command
```bash
pipenv --version
```

If yout dont have it just perform this comman.
```bash
pip install pipenv
```

And when you have installed pipenv you need to activate it. You need to be in the project folder and pefrom this command.
```bash
pipenv shell
```
This command use Pipfile from the project folder and activate virtual environment for the further manipulations with the model.

## Docker Installation

Clone the repo using this command.

```bash
git clone https://github.com/maltsevoff/mlzoomcamp-midterm-project.git
```

Then you need to be shure that you have docker installed. You can check it using this command.
```bash
docker --version
```

If it outputs current version everything is ok, continue. If yout don't have docker you need firstly install it.

Now bild the docker image using Dockerfile from reposity. You need to be in the project folder and perfrom this command in a terminal.
```bash
docker build -t mlzoomcamp-midterm .
```

Then just run this image using this comman. It will perfrom all needed instruction from Dockerfile and run the flask server locally via 9696 port to have acces to model testing. All dependencies are defined in Pipfile which docker uses automatically.
```bash
docker run -it --rm -p 9696:9696 mlzoomcamp-midterm
```

## Test the model 

In the end of **notebook.ipynb** you can find this test code. Here we have random user from data set and request which call local server on port 9696 to get access to the prediction service.
```python
import requests

url = 'http://localhost:9696/predict'

# it is the random customer from the data set.

customer = {
    'age': 58,
    'job': 'management',
    'marital': 'married',
    'education': 'tertiary',
    'default': 'no',
    'balance': 2143,
    'housing': 'no',
    'loan': 'no',
    'contact': 'unknown',
    'day': 5,
    'month': 'may',
    'duration': 1000,
    'campaing': 1,
    'pdays': -1,
    'previous': 0,
    'poutcome': 'unknown',
    'subscribe': 0
}

response = requests.post(url, json=customer).json()

print(response)
```

Just perform this code in **notebook.ipynb** and after running a docker image and see a result of prediction.
