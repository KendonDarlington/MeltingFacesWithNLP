#--------------------------------------wordcloud article-----------------------
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from PIL import Image
import os

#Lets set our working directory. You will need to change this to the root 
#of the folder you downloaded from Github!!
os.chdir('C:\\Users\\kendo\\Documents\\Python Scripts\\Medium\\Phrase Analysis')

#Here is my text. I repeat words i want bigger, the filler words are going to be listed once below
#to make them smaller. They will just be a big ol pile of data science jargon
text = """
melting faces with nlp finding themes in 90s grunge with python. 
kendon kendon kendon kendon kendon 
melting faces melting faces melting faces melting faces
nlp nlp nlp nlp nlp nlp
darlington darlington
melting melting melting
python python python
awesome tutorial 
natural language processing
anaconda 
learn nltk toolkit wordcloud stylecloud data generation 
cool knowlege 90s 90s 90s grunge 

graph chart visualize process spyder jupyter console Anonymization
Algorithm Artificial-Intelligence AI Bayes-Theorem Behavioural 
Big-Data Classification Clickstream analytics Clustering
Governance Mining Set Scientist Decision Trees
Dimension In-Memory Database Metadata Outlier Predictive Modelling
Quantile R Random Forest Standard Deviation Spark Autoregression
Regression Backpropogation Baysean Statistics Binary Boosting
Binomial Convergence Validation Dataset Dashboard Dimensionality
False-Negative Frequency Ngrams Hyperparameter Histogram Hypothesis
Kmeans Keras Logistic Model Multivariate Nominal Distribution Overfitting
Outlier Precision Recall RMSE Supervised Unsupervised
"""

#The mask will by my drawing of a left handed fender mustang. The same type 
#of guitar kurt cobain often used. 
mask = np.array(Image.open( 'mustang.png'))


#Lets make the wordcloud with a hot pink outline, masked by our guitar 
#aka it the words will fill the shape of the guitar
wc = WordCloud(background_color='white', mask=mask, mode='RGB',
               width=1000, max_words=1000, height=1000,
               random_state=1, contour_width=5, contour_color='hotpink')
wc.generate(text)
plt.figure(figsize=(10, 10))
plt.imshow(wc, interpolation='bilinear')
plt.tight_layout(pad=0)
plt.axis('off')
plt.show()


#Mask of the wings that will go behind kurt. 
mask = np.array(Image.open( 'wings.png'))


#Lets make the wordcloud with a hot pink outline, masked by our guitar 
#aka it the words will fill the shape of the guitar
wc = WordCloud(background_color='white', mask=mask, mode='RGB',
               width=7000, max_words=1000, height=4000,
               random_state=1, contour_width=5, contour_color='hotpink')
wc.generate(text)
plt.figure(figsize=(10, 10))
plt.imshow(wc, interpolation='bilinear')
plt.tight_layout(pad=0)
plt.axis('off')
plt.show()
