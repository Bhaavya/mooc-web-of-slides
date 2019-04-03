from PyPDF2 import PdfFileWriter, PdfFileReader
import sys
import argparse

#import pytesseract as pt
#import srt
import os
from collections import Counter
import numpy as np
import re
def convert_pdf_images(inpath,outpath):
  # outpath = outpath+'slides/'
  # if not os.path.exists(outpath):
  #   os.mkdir(outpath)
  # pages = convert_from_path(inpath, 500)
  # i = 0
  # slides_content = []
  # for page in pages:
  #   page.save(outpath+'slide'+str(i)+'.jpg', 'JPEG')
  #   img = cv2.imread(outpath+'slide'+str(i)+'.jpg')
  #   i = i+1
  # return

  inputpdf = PdfFileReader(open(inpath, "rb"))
  for i in range(inputpdf.numPages):
  	output = PdfFileWriter()
  	output.addPage(inputpdf.getPage(i))
  	with open("document-page%s.pdf" % i, "wb") as outputStream:
  		output.write(outputStream)





def split_images(slidespath):

	slidenames = os.listdir(slidespath)
	pathOut = slidespath+'newslides/'
	if not os.path.exists(pathOut):
		os.mkdir(pathOut)
  
	for s in slidenames:
		if('slide' in s):
			number = int(s.split('slide')[1].split('.jpg')[0])
			image = cv2.imread(slidespath+'slides/'+s)
			height, width = image.shape[:2]
			print (image.shape)
			
			# Let's get the starting pixel coordiantes (top left of cropped top)
			start_row, start_col = int(0+height*0.1), int(0+width*0.2)
			# Let's get the ending pixel coordinates (bottom right of cropped top)
			end_row, end_col = int(height * .5), int(width-width*0.2)
			cropped_top = image[start_row:end_row , start_col:end_col]
			cv2.imwrite( pathOut + "slide%d.jpg" %(number*2), cropped_top)

			# Let's get the starting pixel coordiantes (top left of cropped bottom)
			start_row, start_col = int(height * .5), int(0+width*0.2)
			# Let's get the ending pixel coordinates (bottom right of cropped bottom)
			end_row, end_col = int(height-(height*0.1)), int(width-width*0.2)
			cropped_bot = image[start_row:end_row , start_col:end_col]
			cv2.imwrite( pathOut + "slide%d.jpg" % (number*2+1), cropped_bot)

c = '/Users/bhavya/Documents/mooc-web-of-slides-local/high_res_segmented_data/bayesian-methods-in-machine-learning/' 
# root, dirs, files = os.walk(c).next()
root, dirs, files = next(os.walk(c))
lectures = dirs
if c in lectures:
  lectures.remove(c)
lectures = [c+l for l in lectures]
print (lectures)
#lectures = ['data/recommender-systems-introduction/4','data/recommender-systems-introduction/15']
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
	  # split_images(l+'/')
       # save frame as JPEG file
          




