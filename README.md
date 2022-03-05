# University Admission Prediction Using Linear Regression
This project is about using linear regression models to predict admission likelihood of a Masters Programs candidate based on some independent variables such as GRE and TOEFL scores, University Rating, Undergraduate GPA, etc.

The dataset is available at this [*Kaggle data repository*](https://www.kaggle.com/mohansacharya/graduate-admissions) and the paper describing the data is [Acharya et al. 2019](https://ieeexplore.ieee.org/document/8862140).


## Task Requirements
Building linear regression models from scratch **without** using any off-the-shelf linear regression source code or library. Specifically, implement the following gradient descent optimisation algorithms 
1. Standard Gradient Descent 
2. Stochastic Gradient Descent 
3. Mini-batch Gradient Descent  

to minimise the Sum Squared Error (SSE). The aim is to obtain accurate predictive performance on
the test set of 100 observations in the data.

## Instructions
The three gradient descent optimisation algorithms are all implementated into [**math_assignment.py**](https://github.com/LetianLee/University-Admission-Prediction/blob/main/math_assignment.py) file.  
Please launch the code from the command line, use 
```bash
python math_assignment.py
```


**The source code are recommended to read by Jupyter Notebook.  
Please read Jupyter file [**math.ipynb**](https://github.com/LetianLee/University-Admission-Prediction/blob/main/math.ipynb).**   


If you are interested, you can read my Lab Report by the file [**Math_Assignment.pdf**](https://github.com/LetianLee/University-Admission-Prediction/blob/main/Math_Assignment.pdf).

Thanks!
