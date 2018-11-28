
# coding: utf-8

# In[70]:


import json
import os
import codecs
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import *
from pattern.en import parse
import nltk
from nltk.corpus import stopwords
import numpy as np
import ast

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

verb_tags = ['VB', 'VBS','VBG','VBN','VBP','VBZ']


# In[71]:


def read_file(filename):
    json_dict = {}
    with open(filename, 'r', encoding='utf8') as f:
        x = 0
        line = f.readline()
        while line != '':
            while line == '\n':
                line = f.readline()
            line = line.split('\n')[0]
            if line[0] == '\f':
                for y in range(len(line)):
                    if line[y] == '\f':
                        x += 1
                        json_dict['slide' + str(x)] = {}
                    else:
                        break
                if line[1:] == '8/25/2016':
                    f.readline()
                    line = f.readline()
                    json_dict['slide' + str(x)]['title'] = preprocess_string(line)
                else:
                    json_dict['slide' + str(x)]['title'] = preprocess_string(line[1:])
            else:
                processed_string = preprocess_string(line)
                if x == 0:
                    if line == '8/25/2016':
                        f.readline()
                        line = f.readline()
                        processed_string = preprocess_string(line)
                    json_dict['slide' + str(x)] = {}
                    json_dict['slide' + str(x)]['title'] = processed_string
                else:
                    if 'text' not in json_dict['slide' + str(x)]:
                        json_dict['slide' + str(x)]['text'] = processed_string
                    else:
                        json_dict['slide' + str(x)]['text'] += processed_string
            line = f.readline()
    return json_dict


# In[72]:


def read_transcript(slide_content):
    courses = ['bayesian-methods-in-machine-learning', 'cluster-analysis','cs-410',
               'language-processing']
    for course in courses:
        with os.scandir('./slides_augmented_content/' + course) as it:
            for folder in it:
                if not folder.is_file():
                    with os.scandir('./slides_augmented_content/' + course + '/' + folder.name) as transcript_folder:
                        for entry in transcript_folder:
                            if entry.is_file():
                                if entry.name.lower().endswith('.txt'):
                                    try:
                                        with open(entry, 'r') as f:
                                            x = 1
                                            line = f.readline()
                                            slide_num = 'slide0'
                                            while line != '':
                                                if x % 2 == 1 and line != '\n':
                                                    slide_num = line.split('\n')[0][:-4]
                                                elif x % 2 == 0:
                                                    if entry.name in slide_content[course]:

                                                        slide_content[course][entry.name][slide_num]['lecture_transcript'] = array2txt(ast.literal_eval(line))
                                                x += 1
                                                line = f.readline()
                                    except FileNotFoundError:
                                        continue
                        print('Finished adding transcript for', entry)
    return slide_content


# In[73]:


def array2txt(array):
    string = ''
    for word in array:
        string += (word + ' ')
    return preprocess_string(string[:-1])


# Remove all stopwords and lemmatize all words in each line of the slide

# In[74]:


def preprocess_string(string):
    result = ''
    string = string.lower()
    tokenized = nltk.word_tokenize(string)
    for x in range(len(tokenized)):
        if tokenized[x] not in stopwords.words('english'):
            parse_output = parse(tokenized[x], relations=True, lemmata=True).split("/")
            result += parse_output[len(parse_output) - 1].strip() + ' '
    return result[:-1] + '\n'


# In[75]:


def get_course_json(path):
    course = {}
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                course[entry.name] = read_file(entry)
                print('Finished converting file', entry.name)
    return course


# In[76]:


def write_to_json(filename, data):
    with open(filename + '.json', 'w', encoding='utf8') as outfile:
        json.dump(data, outfile)


# In[77]:


def main():
    courses = ['bayesian-methods-in-machine-learning', 'bayesian-statistics', 'cluster-analysis','cs-410',
               'language-processing','ml-clustering-and-retrieval','recommender-systems-introduction', 'text-mining-analytics']
    courses_json = {}
    for course in courses:
        courses_json[course] = get_course_json('./pdftotext/' + course)
    courses_json = read_transcript(courses_json)
    write_to_json('courses_json_preprocessed', courses_json)


# In[78]:


main()


# In[79]:


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
                        f.write('\t\t\t' + content + ':' + data[course][lecture][slide][content] + '\n')
                    f.write('\t\t}\n')
                f.write('\t}\n')
            f.write('}\n')


# In[80]:


make_json_readable('courses_json_preprocessed.json', 'readable_courses_preprocessed.txt')

