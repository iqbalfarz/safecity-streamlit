# SafeCity Sexual Harassment Story Classification

This is the Web Version of final result of "SafeCity Sexual Harassment Story Classification" using Streamlit.

Official website to share your story: [Click Here](https://www.safecity.in)

## Table of Contents
* [Demo](#demo)
* [Overview](#overview)
* [Problem Statement](#problem-statement)
* [Source and Useful Links](#source-and-useful-links)
* [Real-world/business Objectives and Constraints](#real-world-business-objectives-and-constraints)
* [Mapping to Machine Learning Problem](#mapping-to-machine-learning-problem)
* [Model Training](#model-training)
* [Technical Aspect](#technical-aspect)
* [Installation](#installation)
* [Run](#run)
* [Deployment on Heroku](#deployment-on-heroku)
* [Directory Tree](#directory-tree)
* [Future Work](#future-work)
* [Technologies used](#technologies-used)
* [Team](#team)
* [Credits](#credits)
<hr><hr>

## Demo
Link : [https://safecity-streamlit.herokuapp.com/](https://safecity-streamlit.herokuapp.com)

[![](https://imgur.com/FZLGfkn.jpeg)](https://safecity-streamlit.herokuapp.com/)


## Overview
Safecity is a Platform as a Service(PaaS) product that powers communities, police and city government to prevent violence in public and private spaces. SafeCity technology stack collects and analyses crowdsourced, anonymous reports of violent crime, identifying patterns and key insights. 
SafeCity is the largest online platform where people share their pesonal Sexual Harassment stories.

Safecity is an initiative of the Red Dot Foundation based in Washingon DC, U.S.A and its sister concern Red Dot Foundation based in Mumbai, India. Our dataset is the world’s largest, with 25 participating cities/countries/organizations.

## Problem Statement
Classifying the stories shared online among types of Harassment Like: Commenting, Ogling/Staring, Groping/Touching.
This problem is proposed as both **Binary** and **Multi-label Classification**.


## Source and useful links
Data Source: https://github.com/swkarlekar/safecity

YouTube : https://www.youtube.com/channel/UCM8Hln70jUqQpoDz9zPuTIg?sub_confirmation=1

Research paper: https://safecity.in/wp-content/uploads/2019/04/UNC-Chapel-Hill.pdf
### Limitations of research paper:
1. Metrics used in research paper is not good to judge the model performance.
    * Binary Classification: In research paper accuracy is used to judge the model. But, we know that dataset is imbalanced, So, It is good to use precision and recall to judge the model.(don't even use just f1-score)
    * Multi-label classification: In research paper Hamming score and Exact-match(accuracy) is used but it is better to use bothe precision and recall for each label.(can get it by using Classification_report).

Blog : https://medium.com/omdena/exploratory-data-analysis-of-the-worlds-biggest-sexual-harassment-database-107e7682c732

Guide to Machine Learning by Facebook: https://research.fb.com/the-facebook-field-guide-to-machine-learning-video-series/


## Real-World Business Objectives and Constraints
* Low-latency requirement because we need to suggest the tag in runtime over the internet.
* Interpretability is important.
* False Positives and False Negatives may lead to inconsistency to take appropriate action

## Mapping to Machine Learning Problem
1.  ### Data
    a. Data Overview:
    Refer : https://github.com/swkarlekar/safecity
    All of the data is in 2 folders.
    * Folder 1(Binary Classification):
      * commenting_data:
          * train
          * test
          * validation
      * groping_data:
          * train
          * test
          * validation
      * ogling_data:
          * train 
          * test
          * validation
    * Folder 2(Multi-label classification):
      * train 
      * test
      * validation   
       
**This data is for research purposes only and is publicly available at <http://maps.safecity.in/reports>. Please contact SafeCity moderators at <http://maps.safecity.in/contact> for permission before the use of this data. We thank the SafeCity moderators for their assistance with the data download.**

2. ### Types of Machine Learing Problem
    * #### Single-Label Classification 

      * The data for single-label classification is given in two columns, with the first column being the description of the incident and the second column being 1 if the category of sexual harassment is present and 0 if it is not.   

      * Examples from **Groping** Binary Classification Dataset:

        | Description | Category |
        |---|---|
        | Was walking along crowded street, holding mums hand, when an elderly man groped butt, I turned to look at him and he looked away, and did it again after a while.I was 12 yrs old then. | 1 |
        | This incident took place in the evening.I was in the metro when two guys started staring. | 0 |
        | Catcalls and passing comments were two of the ghastly things the Delhi police at the International Airport put me and my friend through. It is appalling that the protectors and law enforcers at the airport can make someone so uncomfortable. |	0 |  

        10% of each dataset was randomly selected and held-out for the test set. From the remaining training data, 10% was randomly selected and set aside for the development set. 

        | Category | % Positive | 
        |---|---|
        | Commenting | 39.3% |
        | Ogling | 21.4% |
        | Groping | 30.1% |

        For each category, there are 7201 training samples, 990 development samples, and 1701 test samples. 

    * #### Multi-Label Classification 

      * The data for multi-label classification is given in four columns, with the first column being the description of the incident and the second, third, and fourth column being 1 if the category of sexual harassment is present and 0 if it is not.   

      * Examples from **Multi-Label** Classification Dataset: 

        | Description | Commenting | Ogling/Facial Expressions/Staring | Touching/Groping |
        |---|---|---|---|
        | Was walking along crowded street, holding mums hand, when an elderly man groped butt, I turned to look at h7m and he looked away, and did it again after a while.I was 12 yrs old then. | 0 |	0 |	1 |
        | This incident took place in the evening.I was in the metro when two guys started staring. |	0 |	1 |	0 |
        | Catcalls and passing comments were two of the ghastly things the Delhi police at the International Airport put me and my friend through. It is appalling that the protectors and law enforcers at the airport can make someone so uncomfortable. | 1 |	1 |	0 |

        10% of the dataset was randomly selected and held-out for the test set. From the remaining training data, 10% was randomly selected and set aside for the development set. 

        | Commenting | Ogling | Groping | Examples in Dataset | 
        |---|---|---|---|
        | 1 | 1 | 1 | 351 | 
        | 1 | 1 | 0 | 819 | 
        | 1 | 0 | 1 | 459 | 
        | 0 | 1 | 1 | 201 | 
        | 1 | 0 | 0 | 2256 | 
        | 0 | 0 | 1 | 1966 | 
        | 0 | 1 | 0 | 743 | 
        | 0 | 0 | 0 | 3097 | 

        There are 7201 training samples, 990 development samples, and 1701 test samples.   

## Model Training
For Model training part [Click Here]()

## Technical aspect
I solved this problem using both Machine Learning and Deep Learning Algorithms
1. ### Machine Learning:
  * Binary Classification:
    * **Encoding**: For text enconding we use TF-IDF(Term-Frequency Inverse-Document Frequency).
    * **ML Algorithms**: 
        * Logistic Regression
        * SVM(linear, 'rbf','ploy')
        * Naive Bayes(Guassian, Multinomial, Binomial)
        * Decision Tree Classifier
        * Random Forest
        * GBDT(Gradient Boosting Decision Tree)
        * Didn't use KNN because of latency problem.
    * **Best Performance Algorithm**: Logistic Regression Won the Game.
    * **Performance Metrics**: 
        * Precision
        * Recall.
    * **Library**: scikit-learn
  * Multilabel Classification:
      *  **Agorithms**: 
          *  BinaryRelevance
          *  OneVsRestClassifier(with LogitsticRegression, etc.)
          *  ClassifierChain
          * LabelPowerset
      *  **Library used**:
          * skmultilearn
      * **Performance Metrics**: 
          * Precision 
          * Recall
          * Hamming score
          * Hamming loss
          * Exact math(accuracy) were used in Research Paper, But exact match is not the right metric to judge how model is performing on each label.
2. ### Deep Learning: 
    In Deep Learning, the only difference between Binary and Multi-label Classification is the last layer of Neural Network.
    * **Encoding**:   
        * Used Embedding Layer to train our own embedding
        * Used Word2Vec pre-trained encoding for each word(50d, and 300d)
    * **DL Algorithms**:
      * CNN: Used Conv1D, 1D convolution is used to preserve sequential relationship at some extent.
      * RNN: Recurrent Neural Network, we used LSTM(Long Short Term Memory).
      * CNN-RNN: Combination of Conv1D and RNN
    * **Miscellaneous Details**:
      * Used Adam Optimizer.
      * Used ReLU activation function as hidden activation function.
      * Saved Best Model according to best validation recall.
      * Used Dropout to overcome overfitting problem.


        

## Installation
The code is written in python==3.9.5. If you want to install python,  [Click here](https://www.python.org/downloads/). Don't forget to upgrade python if you are lower version. upgrade using `pip install --upgrade python` . 
1. Clone the repository
2. install requirements.txt:
    ```
    pip isntall -r requirements.txt
    ```
    
## Run
Run this project using below command on your local machine.
    ```
    streamlit run app.py
    ```    

## Deployment on Heroku
To deploy streamlit project on Heroku. You can follow this video by Krish Naik: [Click Here](https://www.youtube.com/watch?v=IWWu9M-aisA)

## Interpretation
Interpreted both Machine Learning and Deep Learning model using LIME(Local Interpretability Model-agnostic Explanation).
[![Stroy](https://i.imgur.com/bZwZown.jpg)](https://i.imgur.com/bZwZown.jpg)
[![Explanation](https://i.imgur.com/8TrZmf1.jpg)](https://i.imgur.com/8TrZmf1.jpg)

## Directory Tree
```
├── Procfile
├── README.md
├── app.py
├── best_cnn2_embedding_mode.hdf5
├── harass.jpg
├── requirements.txt
├── setup.sh
├── sexualharassment.jpg
├── tokenizer.jpg
```
## Future Work
1. To get more data.
2. Use State-of-the-art algorithms like **BERT**(**B**i-directional **E**ncoding **R**epresentation from **T**ransformer).
3. Try to use different trained model to classify each label.(Commenting, Ogling, Groping)

## Technologies used
[![](https://forthebadge.com/images/badges/made-with-python.svg)](https://www.python.org)
[<img target="_blank" src="https://keras.io/img/logo.png" width=200>](https://keras.io/)
[<img target="_blank" src="https://banner2.cleanpng.com/20180408/jxw/kisspng-tensorflow-deep-learning-keras-machine-learning-ca-thumbtack-5ac9a963b52208.4587965815231655397419.jpg" width=200>](https://www.tensorflow.org/)
[<img target="_blank" src="https://streamlit.io/images/brand/streamlit-logo-primary-colormark-darktext.png" width=200>](https://streamlit.io/)
[<img src="https://www.fullstackpython.com/img/logos/heroku.png" width=200>](https://www.heroku.com/)
[<img src="https://numpy.org/images/logos/numpy.svg" width=200>](https://www.numpy.org)
[<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ed/Pandas_logo.svg/1200px-Pandas_logo.svg.png" width=200>](https://pandas.pydata.org)

## Team
<a href="https://github.com/iqbal786786"><img src="https://avatars.githubusercontent.com/u/32350208?v=4" width=300></a>
|-|
[Muhammad Iqbal Bazmi](https://github.com/iqbal786786) |)

## Credits
1. [Applied AI Course](https://www.appliedaicourse.com) : For teaching in-depth Machine Learning and Deep Learning
2. [Krish Naik](https://www.youtube.com/channel/UCNU_lfiiWBdtULKOw6X0Dig) : For Heroku Developemnt and Github repository management.
3. [SafeCity](https://www.safecity.in/) : For [research paper](https://safecity.in/wp-content/uploads/2019/04/UNC-Chapel-Hill.pdf) and [dataset](https://github.com/swkarlekar/safecity).
