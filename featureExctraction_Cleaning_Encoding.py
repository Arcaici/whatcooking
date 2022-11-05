#this file aim to be a little function library
#for apply cleaning, extraction and encoding to this dataset,
#in this way i avoide code duplication and the scripts are more readable.
#
#main functions: ingredientsExtraction, cleaningIngredients
import re
import pandas as pd
import numpy as np
from sklearn import preprocessing
from nltk.stem import WordNetLemmatizer


def remove_numbers(ing):
    return [[re.sub("\d+", "", x) for x in y] for y in ing]

def make_lowercase(ing):
    return [[x.lower() for x in y] for y in ing]


def remove_extra_whitespace(ing):
    return [[re.sub('\s+', ' ', x).strip() for x in y] for y in ing]


def remove_units(ing):

    units = ['g', 'lb', 's', 'n', 'oz']

    def check_word(word):
        s = word.split()
        resize_data = [word for word in s if word.lower() not in units]

        return ' '.join(resize_data)

    return [[check_word(x) for x in y] for y in ing]


def remove_special_chars(ing):
    # remove certain special characters from ingredients

    ing = [[x.replace("-", " ") for x in y] for y in ing]
    ing = [[x.replace("&", " ") for x in y] for y in ing]
    ing = [[x.replace("'", " ") for x in y] for y in ing]
    ing = [[x.replace("''", " ") for x in y] for y in ing]
    ing = [[x.replace("%", " ") for x in y] for y in ing]
    ing = [[x.replace("!", " ") for x in y] for y in ing]
    ing = [[x.replace("(", " ") for x in y] for y in ing]
    ing = [[x.replace(")", " ") for x in y] for y in ing]
    ing = [[x.replace("/", " ") for x in y] for y in ing]
    ing = [[x.replace("/", " ") for x in y] for y in ing]
    ing = [[x.replace(",", " ") for x in y] for y in ing]
    ing = [[x.replace(".", " ") for x in y] for y in ing]
    ing = [[x.replace(u"\u2122", " ") for x in y] for y in ing]
    ing = [[x.replace(u"\u00AE", " ") for x in y] for y in ing]
    ing = [[x.replace(u"\u2019", " ") for x in y] for y in ing]

    return ing

def applyLemmatizer(ing):
    lemmatize = WordNetLemmatizer()
    ing = [[x.replace(x, lemmatize.lemmatize(x)) for x in y] for y in ing]
    return ing

def ingredientsExtraction(data):
    # This function clean the feature ingredients by varius operations,
    #
    # remove : numbers, special chars, extra white space and units
    # make   : lower case
    # apply  : lemmatizer

    data['ingredients'] = remove_numbers(data['ingredients'])
    data['ingredients'] = remove_special_chars(data['ingredients'])
    data['ingredients'] = make_lowercase(data['ingredients'])
    data['ingredients'] = remove_extra_whitespace(data['ingredients'])
    data['ingredients'] = remove_units(data['ingredients'])
    data['ingredients'] = applyLemmatizer(data['ingredients'])

    s = ''
    for i in range(len(data['ingredients'])):
        s =''
        for x in data['ingredients'].loc[i]:
            x.replace(x, re.sub('[^A-Za-z]+', '', x))
            s = s + " " + x
        data['ingredients'].loc[i] = s
    return data

def cleaningIngredients(data):
    #This function clean the feature ingredients by varius operations
    #
    #remove : numbers, special chars, extra white space and units
    #make   : lower case
    #apply  : lemmatizer

    data['ingredients'] = remove_numbers(data['ingredients'])
    data['ingredients'] = remove_special_chars(data['ingredients'])
    data['ingredients'] = make_lowercase(data['ingredients'])
    data['ingredients'] = remove_extra_whitespace(data['ingredients'])
    data['ingredients'] = remove_units(data['ingredients'])
    data['ingredients'] = applyLemmatizer(data['ingredients'])

    for y in range(len(data['ingredients'])):
       data['ingredients'].loc[y] = [x.replace(" ", "") for x in data['ingredients'].loc[y]]

    return data['ingredients']

def encodeCusine(t):
    #lebal encoder
    le =preprocessing.LabelEncoder()
    t = le.fit_transform(t)

    return t, le

def decodeCusine(t, le):
    #label decoder
    t = le.inverse_transform(t)
    return t