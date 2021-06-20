# SafeCity Sexual Harassment Story Classification

This is the **Web Version** of the final result of "**SafeCity Sexual Harassment Story Classification**" using **Streamlit**.

I used the **best algorithm** found after training different models for deployment. I used **LIME** to interpret the model.

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
Safecity is a Platform as a Service(PaaS) product that powers communities, police, and city government to prevent violence in public and private spaces. SafeCity technology stack collects and analyses crowdsourced, anonymous reports of violent crime, identifying patterns and key insights. 
SafeCity is the largest online platform where people share their personal Sexual Harassment stories.

Safecity is an initiative of the Red Dot Foundation based in Washington DC, U.S.A, and its sister concern Red Dot Foundation based in Mumbai, India. Our dataset is the world’s largest, with 25 participating cities/countries/organizations.

## Problem Statement
Classifying the stories shared online among types of Harassment Like Commenting, Ogling/Staring, Groping/Touching.
This problem is proposed as both **Binary** and **Multi-label Classification**.


## Source and useful links
Data Source: https://github.com/swkarlekar/safecity

YouTube: https://www.youtube.com/channel/UCM8Hln70jUqQpoDz9zPuTIg?sub_confirmation=1

Research paper: https://safecity.in/wp-content/uploads/2019/04/UNC-Chapel-Hill.pdf

Blog: https://medium.com/omdena/exploratory-data-analysis-of-the-worlds-biggest-sexual-harassment-database-107e7682c732

Guide to Machine Learning by Facebook: https://research.fb.com/the-facebook-field-guide-to-machine-learning-video-series/


## Real-World Business Objectives and Constraints
* Low-latency requirement because we need to suggest the tag in runtime over the internet.
* Interpretability is important.
* False Positives and False Negatives may lead to inconsistency to take appropriate action

## Mapping to Machine Learning Problem
* The whole training part is present in different repository: [Click Here](https://github.com/iqbal786786/safecity-training)

## Model Training
For the Model training part: [Click Here](https://github.com/iqbal786786/safecity-training)

## Technical aspect
For technical details, go to Model training part: [Click Here](https://github.com/iqbal786786/safecity-training) 

## Installation
The code is written in python==3.9.5. If you want to install python,  [Click here](https://www.python.org/downloads/). Don't forget to upgrade python if you are using a lower version. upgrade using `pip install --upgrade python`. 
1. Clone the repository
2. install requirements.txt:
    ```
    pip install -r requirements.txt
    ```
    
## Run
Run this project using the below command on your local machine.
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
2. Tune the model with different values of hyper-parameters.(It is tedious task)
3. Use state-of-the-art algorithms like **BERT**(**B**i-directional **E**ncoding **R**epresentation from **T**ransformer).

## Technologies used
[![](https://forthebadge.com/images/badges/made-with-python.svg)](https://www.python.org)
[<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Scikit_learn_logo_small.svg/1200px-Scikit_learn_logo_small.svg.png" width=200 height=100>](https://scikit-learn.org/stable/)
[<img target="_blank" src="https://keras.io/img/logo.png" width=200>](https://keras.io/)
[<img target="_blank" src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Tensorflow_logo.svg/1200px-Tensorflow_logo.svg.png" width=200>](https://www.tensorflow.org/)
[<img target="_blank" src="https://streamlit.io/images/brand/streamlit-logo-primary-colormark-darktext.png" width=200>](https://streamlit.io/)
[<img src="https://www.fullstackpython.com/img/logos/heroku.png" width=200>](https://www.heroku.com/)
[<img src="https://numpy.org/images/logos/numpy.svg" width=200>](https://www.numpy.org)
[<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ed/Pandas_logo.svg/1200px-Pandas_logo.svg.png" width=200>](https://pandas.pydata.org)

## Team
<a href="https://github.com/iqbal786786"><img src="https://avatars.githubusercontent.com/u/32350208?v=4" width=300></a>
|-|
[Muhammad Iqbal Bazmi](https://github.com/iqbal786786) |)

## Credits
1. [Applied AI Course](https://www.appliedaicourse.com): For teaching in-depth Machine Learning and Deep Learning
2. [Krish Naik](https://www.youtube.com/channel/UCNU_lfiiWBdtULKOw6X0Dig): For Heroku Developmnt and Github repository management.
3. [SafeCity](https://www.safecity.in/) : For [research paper](https://safecity.in/wp-content/uploads/2019/04/UNC-Chapel-Hill.pdf) and [dataset](https://github.com/swkarlekar/safecity).
