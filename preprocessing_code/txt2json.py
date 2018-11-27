
# coding: utf-8

# In[10]:


import json
import os
import codecs
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import *
from pattern.en import parse
import nltk
from nltk.corpus import stopwords
import numpy as np

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

verb_tags = ['VB', 'VBS','VBG','VBN','VBP','VBZ']


# In[11]:


def read_file(filename):
    json = {}
    with open(filename, 'r', encoding='utf8') as f:
        x = 1
        line = f.readline()
        while line != '':
            while line == '\n':
                line = f.readline()
            line = line.split('\n')[0]
            if line[0] == '\f':
                x += 1
                json['slide' + str(x)] = {}
                if line[1:] == '8/25/2016':
                    f.readline()
                    line = f.readline()
                    json['slide' + str(x)]['title'] = preprocess_string(line)
                else:
                    json['slide' + str(x)]['title'] = preprocess_string(line[1:])
            else:
                processed_string = preprocess_string(line)
                if x == 1:
                    if line == '8/25/2016':
                        f.readline()
                        line = f.readline()
                        processed_string = preprocess_string(line)
                    json['slide' + str(x)] = {}
                    json['slide' + str(x)]['title'] = processed_string
                else:
                    if 'text' not in json['slide' + str(x)]:
                        json['slide' + str(x)]['text'] = processed_string
                    else:
                        json['slide' + str(x)]['text'] += processed_string
            line = f.readline()
    return json


# Remove all stopwords and lemmatize all words in each line of the slide

# In[12]:


def preprocess_string(string):
    result = ''
    string = string.lower()
    tokenized = nltk.word_tokenize(string)
    for x in range(len(tokenized)):
        if tokenized[x] not in stopwords.words('english'):
            parse_output = parse(tokenized[x], relations=True, lemmata=True).split("/")
            result += parse_output[len(parse_output) - 1].strip() + ' '
    return result[:-1] + '\n'


# In[13]:


def get_course_json(path):
    course = {}
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                course[entry.name] = read_file(entry)
                print('Finished converting file', entry.name)
    return course


# In[14]:


def write_to_json(filename, data):
    with open(filename + '.json', 'w', encoding='utf8') as outfile:
        json.dump(data, outfile)


# In[15]:


def main():
    courses = ['bayesian-methods-in-machine-learning', 'cluster-analysis','cs-410',
               'language-processing','ml-clustering-and-retrieval','recommender-systems-introduction']
    courses_json = {}
    for course in courses:
        courses_json[course] = get_course_json('./pdftotext/' + course)
    write_to_json('courses_json_preprocessed', courses_json)


# In[16]:


main()


# In[17]:


def make_json_readable(jsonname, filename):
    data = {}
    with open(jsonname, 'r', encoding='utf8') as f:
        data = json.load(f)
    with open(filename, 'w', encoding='utf8') as f:
        f.write('')
    with open(filename, 'a', encoding='utf8') as f:
        for course in data:
            f.write(course + '{\n')
            for lecture in data[course]:
                f.write('\t' + lecture + '{\n')
                for slide in data[course][lecture]:
                    f.write('\t\t' + slide + '{\n')
                    for content in data[course][lecture][slide]:
                        try:
                            f.write('\t\t\t' + content + ':' + data[course][lecture][slide][content] + '\n')
                        except UnicodeEncodeError:
                            f.write('\t\t\t' + content + ':' + data[course][lecture][slide][content] + '\n')
                    f.write('\t\t}\n')
                f.write('\t}\n')
            f.write('}\n')


# In[18]:


make_json_readable('courses_json_preprocessed.json', 'readable_courses_preprocessed.txt')

