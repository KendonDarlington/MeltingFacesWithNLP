#Import Libraries
import pandas as pd
import nltk 
from nltk import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import stylecloud
from IPython.display import Image
import numpy as np
import matplotlib.pyplot as plt
import os

#You may need to run these two commands after installing NLTK. 
#nltk.download('stopwords')
#nltk.download('punkt')

#This is the main thing you want to change. It will set the working 
#directory and allow you to use relative file paths. Just create a folder to 
#You can see your drop everything into and point this line at that folder. Put the "Grunge Lyrics.xlsx file into this folder
#current working directory by running this in the console: os.getcwd()
os.chdir('C:\\PathToTheFolderYouCreated\\Phrase Analysis')
   
#Load the grunge lyrics into a pandas dataframe
dfAlbums = pd.read_excel('Grunge Lyrics.xlsx')


#Create an empty dataframe to store our results
dfNgramFrequency = pd.DataFrame(columns = ['Band', 'Phrase', 'Frequency'])


#Loop through the bands in and build our df
for index, row in dfAlbums.iterrows():
    
    #This limits our dataframe to just one band at a time
    band = dfAlbums.loc[dfAlbums['band'] == row[0]]['band'].item()
    
    #Get the lyrics column for our current band and store it in a string variable
    lyrics = dfAlbums.loc[dfAlbums['band'] == row[0]]['lyrics'].item()
    
    #Use regex to remove punctuation. 
    lyrics = re.sub(r'[^\w\s]', '', lyrics)
            
    #Lets remove some common stopwords
    stopWords = set(stopwords.words('english'))
    stopWords.add("said")
    
    #Tokenizers break the text into a list of words
    wordTokens = word_tokenize(lyrics) 
    
    
    #VCERIFY by comparing lists? or swapping cloud
    #This will take our bands lyrics, and write all non stopwords to a new list
    sentenceNoStopwords = [w for w in wordTokens if not w.lower() in stopWords]    
    
    #Define the number of words per phrase and put them in a list
    listOGrams = []
    n = 3
    
    #Write our three word phrases to our listOGrams list
    gramGenerator = ngrams(sentenceNoStopwords, n)    
    for grams in gramGenerator:
      listOGrams.append(grams[0] + ' ' + grams[1] + ' ' + grams[2])
      
    #Create a dataframe from list
    df = pd.DataFrame(listOGrams, columns = ['Phrase'])    
    
    #Group by phrase and count the frequency of phrase occurance. Fool around with the index a bit cuz pandas
    df = df.groupby(['Phrase']).size()
    df = df.to_frame()
    df['Phrase'] = df.index
    df.reset_index(drop=True, inplace=True)
    
    #Rename a column, then sort descending
    df = df.rename(columns = {0: "Frequency"})
    df = df.sort_values(by = 'Frequency', ascending = False )    
    
    #Add a column to df for band
    df['Band'] = band
    
    #Append our current dataframe to the one we defined outside of the loop
    dfNgramFrequency = dfNgramFrequency.append(df, ignore_index = True)

#Nirvana
dfNirvana = dfNgramFrequency.loc[dfNgramFrequency['Band'] == 'Nirvana'][['Phrase', 'Frequency']]
dfNirvana.to_csv('Nirvana.csv', index = False)

stylecloud.gen_stylecloud(file_path = 'Nirvana.csv',
                          icon_name='fab fa-python',
                          palette='colorbrewer.diverging.Spectral_11',
                          background_color='black',
                          gradient='horizontal',
                          size=1024)
Image(filename="./stylecloud.png", width=1024, height=1024)


#Smashing Pumpkins
dfSmashingPumpkins = dfNgramFrequency.loc[dfNgramFrequency['Band'] == 'Smashing Pumpkins'][['Phrase', 'Frequency']]
dfSmashingPumpkins.to_csv('SmashingPumpkins.csv', index = False)

stylecloud.gen_stylecloud(file_path = 'SmashingPumpkins.csv',
                          icon_name='fab fa-r-project',
                          palette='colorbrewer.diverging.Spectral_11',
                          background_color='black',
                          gradient='horizontal',
                          size=1024)
Image(filename="./stylecloud.png", width=1024, height=1024)


#Pearl Jam
dfPearlJam = dfNgramFrequency.loc[dfNgramFrequency['Band'] == 'Pearl Jam'][['Phrase', 'Frequency']]
dfPearlJam.to_csv('PearlJam.csv', index = False)

stylecloud.gen_stylecloud(file_path = 'PearlJam.csv',
                          icon_name='fas fa-glasses',
                          palette='colorbrewer.diverging.Spectral_11',
                          background_color='black',
                          gradient='horizontal',
                          size=1024)
Image(filename="./stylecloud.png", width=1024, height=1024)


