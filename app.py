# https://discuss.streamlit.io/t/cache-keras-trained-model/2398

import numpy as np 
import pandas as pd
import tensorflow as tf
import streamlit as st
import streamlit.components.v1 as cm

import pickle
import tensorflow_addons as tfa
from tensorflow.keras.preprocessing.sequence import pad_sequences
from lime.lime_text import LimeTextExplainer # to showcase the Explanation of hte model.


#################################################################
################ Visualization functions LIME ###################
def deep_multi_label_explainer(model, tokenizer, input_text, labels, num_features=None, num_samples=None, bow=True):
    maxlen=100
    results = [] # to store html visualizations for each label
    # labels - all outputs
    for label in labels:
        class_names = ['Non-{}'.format(label), label]

        def make_classifier_pipeline(label=label):
            label_index = labels.index(label)
            # pick the corresponding output node 
            def lime_explainer_pipeline(texts):  
                input_text = pd.Series(texts)# typecasting to series from raw text
                input_sequence = tokenizer.texts_to_sequences(input_text) #changing text into sequences
                padded_input_sequence = pad_sequences(input_sequence, maxlen=maxlen, padding='post') # padding the sequence
                predicted_prob = model.predict(padded_input_sequence)
                prob_true = predicted_prob[:, label_index]
                result = np.transpose(np.vstack(([1-prob_true, prob_true])))  
                result  = result.reshape(-1, 2)

                return result
        
            return lime_explainer_pipeline
       
       # make a classifier function for the required label
        classifier_fn = make_classifier_pipeline(label=label)

        explainer = LimeTextExplainer(
                                        class_names=class_names, # class labels(negative, positive)
                                        kernel_width=25,
                                        bow=bow
        )
        exp = explainer.explain_instance(
            input_text, # explain instance on this particular query text
            classifier_fn, # pipeline
         )
        
        results.append(exp)
    return results
#####################################################################
#################################################################
### load model and preprocess the query to get the prediction ###

# load tokenizer and cache it.
@st.cache(suppress_st_warning=True,allow_output_mutation=True) 
def load_tokenizer():
    with open('tokenizer.pickle', 'rb') as tok:
        tokenizer = pickle.load(tok)
    return tokenizer

# load model and cache it.
@st.cache(allow_output_mutation=True)
def load_dl_model():
    dependencies ={
    'HammingLoss':tfa.metrics.HammingLoss(mode='multilabel',threshold=0.5)
    }
    model = tf.keras.models.load_model('best_cnn2_embedding_recall.hdf5',custom_objects=dependencies)
    
    return model
#################################################################


######################################################################################################
##################  Prediction #######################################################################
# prediction button to predict the category of the Story among Commenting, Ogling, and Groping.

def main():
    # title of the project
    st.title("SafeCity Sexual Harassment Story Classification")

    # label image of the project
    st.image("harass.jpg",caption='Say NO!, Sexual Harassment')

    # print the message
    st.write("Official website to share your story is www.safecity.in.")
    st.write('<a href="www.safecity.in">SafeCity Official</a>',unsafe_allow_html=True)

    # getting the story in text area
    text = st.text_area("Write Your Story Here...")

    prediction=None
    
    if st.button("Share Your Story...Click Here!"):
        with st.spinner('Calculating...'):
            tokenizer = load_tokenizer()
            tokens = tokenizer.texts_to_sequences(pd.Series(text))
            query = pad_sequences(tokens, padding='post', maxlen=100)
            model = load_dl_model()
            st.write("Your story shared.")
            prediction = (model.predict(query))[0]
            st.write('Commenting : {:.2f}'.format(prediction[0]))
            st.write('Ogling     : {:.2f}'.format(prediction[1]))
            st.write('Groping    : {:.2f}'.format(prediction[2]))
            
            st.write("<hr>",unsafe_allow_html=True)
            
            label= ['Commenting','Ogling','Groping']
            vizs = deep_multi_label_explainer(model, tokenizer, text, labels=label)

            cm.html(vizs[0].as_html())          
            cm.html(vizs[1].as_html())
            cm.html(vizs[2].as_html())

#calling main function
if __name__=='__main__':
    main()