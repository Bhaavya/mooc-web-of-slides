from pdf2image import convert_from_path
import sys
import argparse

import cv2
print(cv2.__version__)
#import pytesseract as pt
#import srt
import os
from collections import Counter
import numpy as np
import re
def convert_pdf_images(inpath,outpath):
  outpath = outpath+'slides/'
  if not os.path.exists(outpath):
    os.mkdir(outpath)
  pages = convert_from_path(inpath, 500)
  i = 0
  slides_content = []
  for page in pages:
    page.save(outpath+'slide'+str(i)+'.jpg', 'JPEG')
    img = cv2.imread(outpath+'slide'+str(i)+'.jpg')
    i = i+1
  return 


def extractImages(pathIn, pathOut):
    pathOut = pathOut+'frames/'
    if not os.path.exists(pathOut):
      os.mkdir(pathOut)
  
    count = 0
    vidcap = cv2.VideoCapture(pathIn)
    success,image = vidcap.read()
    success = True
    while success:
      vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*2000))    # added this line 
      success,image = vidcap.read()
      if((count+1)%100==0):
        print ('Read a new frame: ', success)
      if(image is not None):
        height,width,channels = image.shape
        if (height>0 and width>0):
          image = cv2.resize(image, (5000,2813))
          cv2.imwrite( pathOut + "frame%d.jpg" % count, image)     # save frame as JPEG file
          count = count + 1
    return

courses = ['data/ml-clustering-and-retrieval/']
toremove = []
for c in courses:
  #root, dirs, files = os.walk(c).next()
  root, dirs, files = next(os.walk(c))
  lectures = dirs
  if c in lectures:
    lectures.remove(c)
  lectures = [c+l for l in lectures]
  for r in toremove:
    lectures.remove(r)
  print (lectures)
  for l in lectures:
    pdf_inpath = ''
    video_inpath = ''
    contents = os.listdir(l)
    for s in contents: 
      if('.pdf' in s):
        pdf_inpath = l+'/'+s
      else:
        if ('.mp4' in s):
          video_inpath = l+'/'+s
    if (pdf_inpath != ''):
      print (pdf_inpath)
      convert_pdf_images(pdf_inpath,l+'/')
    if (video_inpath!=''):
      print (video_inpath)
      extractImages(video_inpath,l+'/')






