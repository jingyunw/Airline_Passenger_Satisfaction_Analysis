# Airline Passenger Satisfaction Analysis
Author: [JingYun (Jonna) Wang](/jingyunwang24@gmail.com)
<img src="images/aviation.jpeg" style="width:600px;height:400px"/>

## Overview
This project uses exploratory data analysis and 7 classification modelings to predict airline passenger satisfaction after the flight journey. The data contains various survey ratings regarding different customer classes and travel types. Airlines can use my model to better understand the critical features that strongly impact satisfaction.

## Business Problem
Working for a US airline, my job is to create the best classification model to predict whether customers will be satisfied after flight journey and provide recommendations for stakeholders to better understand the critical features that have stronger impact on satisfaction and determine features can be improved for targeting towards specific groups.

***
### Question to Consider:
<b>Q1</b>. What is the satisfaction and dissatisfaction rate?
<br><b>Q2</b>. What are the <b>critical features</b> that have stronger impact on satisfaction rate?
<br><b>Q3</b>. Which <b>class</b> has the highest dissatisfaction rate?
<br><b>Q4</b>. Which <b>flight haul</b> has the highest dissatisfaction rate?
<br><b>Q5</b>. Which <b>age group</b> has the highest dissatisfaction rate?

## Data
The datasets acquired from [Kaggle](https://www.kaggle.com/teejmahal20/airline-passenger-satisfaction) contain customers' satisfaction (target variable) and various survey ratings regarding different customer classes and travel types. 

## Methods
This project use classification modeling to predict satisfaction and train-test-split to evaluate machine learning performance.

## Results

<img src="images/satisfaction pie.png">

|   | Critical Features |
| :---: | :---: |
| Top 1 | Loyal Customers|
| Top 2 | Young (age <20)|
| Top 3 | Inflight entertainment|
| Top 4 | On-board service |
| Top 5 | Seat comfort |

<br>

|   | Highest Dissatisfaction | Low Rating Features |
| :---: | :---: | :---: |
| Class | Eco | online booking, wifi, online boarding|
| Flight Haul | Short | wifi, online booking, gate location |
| Age  | Adult (age 20~40) | wifi, online booking, gate location |

## Model Evaluation
<img src="images/classifier vs accuracy.png">

|  RandomForest | Test Accuracy |
| :---: | :---: |
| default parameter | 96.17% |
| hyperparameter tunning | 97.27%| 


## Conclusion

- Among analyze 129,487 data, a RandomForest Classifier model was selected among 7 different classifiers. After hyperparameter tunning, my model testing accuracy reaches 96.27%.
- Top 5 features that have greater impact on satisfaction are <B>Loyal Customers</b>, <b>Young</b>, <b>Inflight entertainment</b>, <b>On-board service</b> and <b>Seat comfort</b>.
- Customers in <ins>Eco Class</ins>, <ins>Short Haul flight</ins> and <ins>Adult group</ins> have the highest dissatisfaction rate. They gave less rating of 4 and 5 on <b>Ease of Online booking</b>, <b>Infight wifi service</b> and <b>Online Boarding</b> and <b>Gate location</b>
***

### Some recommendations for airline stakeholders are:
<br>*Survey Features*</br>
- <b>Technology</b>: Introducing or improving mobile apps which can benefits all types of customers
    - Easy and fast mobile booking and online check-in can save time and avoid long queue
    - Navigation for local airport gate location
    - Keep customers informed for status changed

- <b>Collaboration</b>: Foster collaboration with airport internal department
    - Negotiate for shorter distance gate location 

## Future Work
Further analysis can be explored on the following to provide additional insights and improve the performance of the model.
- <b>Health and Safety Regulation</b>: Especially during the epidemic, how well the airline deal with social distance and temperature checking can also impact satisfaction
- <b>Customer Service</b>: Analyze online / phone customer service to see the efficiency of support team which can also help increase satisfaction
- <b>Competitive Airline</b>: Acquire data from competitive airlines to understand their strengths and weakness in relation to our airline which can help us refine strategy and increase customer loyalty

## For More Information
See the full analysis and modeling in the [Jupyter Notebook](./Airline-Passenger-Satisfaction.ipynb) and [presentation](./Satisfaction-Presentation.pdf).
For additional information please contact, JingYun (Jonna) Wang at jingyunwang24@gmail.com

## Repository Structure
```
├── data
├── images
├── Airline-Passenger-Satisfaction.ipynb
├── README.md
├── Satisfaction-Presentation.pdf
└── eva_f.py
```