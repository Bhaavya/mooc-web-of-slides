from ast import literal_eval
import sys
import argparse
import pytesseract as pt
import srt
import os
from collections import Counter
import numpy as np
import re
#mapp srt to slides using frame_mappings
def map_script_slides(data_dir, srtfile, frame_mappings_file):
  f = open(frame_mappings_file,'r')
  lines = f.readlines()
  lines = [literal_eval(l.strip()) for l in lines]
  frame_mapping = {}
  for l in lines:
    frame_mapping[l[0]] = l[1]
  srt_parse  = srt.parse(open(srtfile, 'r').read().decode('utf-8'))
  srt_parse = list(srt_parse)
  map_srt_slides = []
  for s in srt_parse:
    #print (s)
    srt_start_time = s.start.seconds
    srt_end_time = s.end.seconds
    start_time =  srt_start_time + (srt_start_time%2)
    end_time = srt_end_time + (srt_end_time%2)
    #group srt for min 10 seconds..
    
    #get the frames within these seconds
    frame_seconds = range(start_time,(end_time+2),2)
    #print frame_seconds
    required_frames = ['frame'+str(i/2)+'.jpg' for i in frame_seconds]
    mid_frame_index = int(len(frame_seconds)/2)  
    map_srt_slides += [(s,frame_mapping[required_frames[mid_frame_index]])]
  target = open(data_dir+'script_mapping.txt','w')
  slide_aug_content = {}
  for m in map_srt_slides:
    augmented_content = m[0].content.strip('\n').split(" ")
    if (m[1] in slide_aug_content):
        slide_aug_content[m[1]] += augmented_content
    else:
        slide_aug_content[m[1]] = augmented_content
    target.write(srt.compose([m[0]]).encode('utf-8'))
    target.write(str(m[1])+'\n')
  target.flush()
  target.close()
  target = open(data_dir+'slide_augmented_content.txt','w')
  slide_aug_content = sorted(slide_aug_content.items(), key=lambda l: l[0])
  for m in slide_aug_content:
    target.write(m[0]+'\n')
    target.write(str(m[1])+'\n')
  target.flush()
  target.close()

#data_dir = 'old_data/cs-410/02_week-1/02_week-1-lessons/'
#srtfile_path = data_dir + '01_lesson-1-1-natural-language-content-analysis.en.srt'
#map_script_slides(data_dir, srtfile_path,data_dir+'frame_mappings.txt')

courses = ['data/recommender-systems-introduction/']
toremove1 = ['data/language-processing/32', 'data/language-processing/35', 'data/language-processing/34', 'data/language-processing/33', 'data/language-processing/20', 'data/language-processing/18', 'data/language-processing/27', 'data/language-processing/9', 'data/language-processing/11', 'data/language-processing/7', 'data/language-processing/29', 'data/language-processing/16', 'data/language-processing/42', 'data/language-processing/6', 'data/language-processing/28', 'data/language-processing/17', 'data/language-processing/1', 'data/language-processing/10', 'data/language-processing/19', 'data/language-processing/26']

lectures = open(courses[0]+'/failed_mappings.txt','r')
lectures =literal_eval(lectures.read())

'''
for c in courses:
  #root, dirs, files = os.walk(c).next()
  root, dirs, files = next(os.walk(c))
  lectures = dirs
  if c in lectures:
    lectures.remove(c)
  lectures = [c+l for l in lectures]
  #for r in toremove1:
  #  lectures.remove(r)
  for r in toremove:
    lectures.remove(r[:-1])
  print (lectures)
print lectures
'''
for data_dir in lectures:
  print (data_dir)
  contents = os.listdir(data_dir)
  for s in contents:
    if('.srt' in s):
      srtfile_path = s
      if ('frame_mappings_new.txt' in contents):
        map_script_slides(data_dir, data_dir+srtfile_path,data_dir+'frame_mappings_new.txt')
      break

