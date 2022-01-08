# Import Required Packages
import sys
import csv
import re
import string
import pickle
from collections import defaultdict
from nltk.corpus import stopwords
from nltk import word_tokenize
STOPWORDS = set(stopwords.words('english'))
from bs4 import BeautifulSoup



# Printing a sample
def print_plot(df,index):
    example = df[df.index == index+1][['text', 'label']].values[0]
    if len(example) > 0:
        print(example[0])
        print('text:', example[1])

# PreProcessing ###
#  load stopwords
def get_stopwords():
    try:
        with open(PATH_STOPWORDS_TR, 'rb') as data_file:
            return pickle.load(data_file)
    except IOError as exc:
        raise IOError("No such stopwords file! Error: " + str(exc))


#  load blacklist
def get_blacklist():
    try:
        with open(PATH_BLACKLIST_TR, 'rb') as data_file:
            return pickle.load(data_file)
    except IOError as exc:
        raise IOError("No such stopwords file! Error: " + str(exc))


#  cleaning stopwords
def clean_stopwords(content):
    # content: str
    content = content.split(" ")
    filtered_list = []
    stopwords = get_stopwords()
    for word in content:
        if word not in stopwords:
            filtered_list.append(word)

    text = ' '.join(filtered_list)
    return text


#  cleaning blacklist
def clean_blacklist(content):
    # content: str
    # return: str
    content = content.split(" ")
    filtered_list = []
    blacklist = get_blacklist()
    for word in content:
        if word not in blacklist:
            filtered_list.append(word)

    text = ' '.join(filtered_list)
    return text


#  cleaning URLs
def clean_url(content):
    #  content: str
    #  return: str
    reg_url = r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))'''
    pattern_url = re.compile(reg_url)
    content = pattern_url.sub('', content)
    return content


#  cleaning e-mails
def clean_email(content):
    reg_email = '\S*@\S*\s?'
    pattern_email = re.compile(reg_email)
    content = pattern_email.sub('', content)
    return content


#  cleaning punctuation
def clean_punctuation(content):
    # regex = re.compile('[%s]' % re.escape(string.punctuation))
    # content = regex.sub(" ", content)
    content = content.translate(content.maketrans("", "", string.punctuation))
    return content


#  cleaning digits
def clean_numbers(content):
    remove_digits = str.maketrans('', '', string.digits)
    text = content.translate(remove_digits)
    return text


def listToString(text):
    #  text: string
    #  return: string
    str1 = " "
    return (str1.join(text))


#  cleaning postfix
def clean_postfix(content):
    #  content: list
    #  return: str
    spesific_punctation = [39, 8217]  # ascii codes of spesific punctations
    all_words = defaultdict(list)
    counter = 0
    for i, word in enumerate(content):
        if len(word) > 2:
            if chr(8217) in word:
                replaced_word = word.replace(chr(8217), " ")  # clean for ’
                replaced_word = replaced_word.split(" ")[0]
                all_words[i].append(replaced_word)
            elif chr(39) in word:
                replaced_word = word.replace(chr(39), " ")  # clean for '
                replaced_word = replaced_word.split(" ")[0]
                all_words[i].append(replaced_word)
            else:
                all_words[i].append(word)
    all_words = sorted(all_words.items())

    text = ""
    for i in range(len(all_words)):
        text = text + " " + all_words[i][1][0]

    return text


def clean_text(content):
    #  text: string
    #  return: string
    content = content.lower()
    cleaned_email = clean_email(content)
    cleaned_email_url = clean_url(cleaned_email)
    cleaned_email_url = listToString(cleaned_email_url.split("."))
    cleaned_email_url = cleaned_email_url.split(" ")
    cleaned_email_url_postfix = clean_postfix(cleaned_email_url)
    cleaned_email_url_postfix_punctuation = clean_punctuation(cleaned_email_url_postfix)
    cleaned_email_url_postfix_punctuation_numbers = clean_numbers(cleaned_email_url_postfix_punctuation)
    cleaned_email_url_postfix_punctuation_numbers_stopwords = clean_stopwords(
        cleaned_email_url_postfix_punctuation_numbers)
    cleaned_email_url_postfix_punctuation_numbers_stopwords_blacklist = clean_blacklist(
        cleaned_email_url_postfix_punctuation_numbers_stopwords)

    filtered_sentence = []
    for word in cleaned_email_url_postfix_punctuation_numbers_stopwords_blacklist.split(" "):
        if len(word) > 2:
            filtered_sentence.append(word)

    text = ' '.join(filtered_sentence)
    return text


### Applying Text Cleanup Preprocesses for Train and Test dataset ###