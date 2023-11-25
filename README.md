# Cultural Classification of Jokes
EECS 486 Final Project by Hongxi Pu, Yichen Zhong, Tianjie Qiu, Zerui Bao, Zhiyuan Sun.

***

### Description of Project:
This is a joke classifier that can classify jokes by their original countries. Currently, the classifier supports six countries: Australia, UK, US, China, Spain and Russia. The last three requires translation to English.

***

##### Datasets:
In directories Datasets, there are Australia/, British/, US/, Chinese/, Spanish/, Russian/

For the first three, these are Australia, UK, and US jokes.
For the last three, these are Chinese, Spainish, and Russian jokes after the translation.

***

##### Model:

In directories Model_code, there are mainly two code for model.

For lstm.py, it is the code for constructing lstm classifier.

For main.ipynb, it is the code for constructing Random Forest, Linear SVC, Logistic Regression and Naive Bayes classifiers.

***

##### 

##### How To Use:

Install necessary packages using "pip install". Below, please find some packages you may need to install:
```sh
pip install bs4
pip install deep_translator
pip install scikit-learn
```
##### Running Software:
Run Jupyter in ./code/main.ipynb
Follow the text hint in ipynb file


##### Additional files:
./code/usdata_clean.py is used to clean the prefix Q:, A: in the original data set
In ./Datasets/test there are 6 folders with test data for each language. Test data are hidden from training.
