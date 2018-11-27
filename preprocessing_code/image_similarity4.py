import numpy as np
from PIL import Image
import os
from scipy import spatial
import indicoio
import time
import math
from ast import literal_eval
import random
indicoio.config.api_key = '79d3cf025a7d9682c128ead5027258ea'
def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

def compare_images(imageA, imageB):
	# compute the mean squared error and structural similarity
	# index for the images
	m = mse(imageA, imageB)
	s = ssim(imageA, imageB)
 	
	return (m,s)


def map_frame_to_slide1(data_dir):
	slidespath = data_dir+'slides/'
	framespath = data_dir+'frames/'
	slidenames = os.listdir(slidespath)
	framenames = os.listdir(framespath)
	l = len(framenames)
	new_framenames = []
	for i in range(l):
		new_framenames +=['frame'+str(i)+'.jpg']
	framenames = new_framenames
	l = len(slidenames)
	new_slidenames = []
	for i in range(l):
		new_slidenames +=['slide'+str(i)+'.jpg']
	slidenames = new_slidenames
	print (slidenames)

	#print (srt_parse)

	slides = []
	for s in slidenames:
		slides += [Image.open(slidespath+s)]
		
	slides = indicoio.image_features(slides)
	slides = zip(slidenames,slides)
	print ("reading slides done")

	frames = []
	i=0
	for f in framenames:
		i = i+1
		frames += [(framespath+f)]
		if(i%100) ==0:
			print(i)

	frame_features = []
	for i in range(0,len(frames),20):
		frame_features +=  indicoio.image_features(frames[i:(i+20)])
		print ((i+20), " done")
	frames = zip(framenames,frame_features)
	print ("reading frames done")



	prev_mapping = 'slide0.jpg'
	prev_max = 0
	max_index = 0
	#map each frame to a slide:

	output = []
	for f in frames:
		second_max = 0
		second_mapping = 'slide0.jpg'
		max_sim = 0
		mapping = 'slide0.jpg'
		for s in slides:
			s1 = 1 - spatial.distance.cosine(f[1],s[1])
			print (s[0],s1)
			if(s1 > max_sim):
				second_max = max_sim
				second_mapping = mapping
				max_sim = s1
				mapping = s[0]
		if(max_sim<0.50):
			mapping = prev_mapping
			max_sim = prev_max
		else:
			if(abs(max_sim-second_max)<0.05):
				print max_sim
				print second_max
				first_max_index = int(mapping.split('slide')[1].split('.jpg')[0])
				second_max_index = int(second_mapping.split('slide')[1].split('.jpg')[0])
				first_difference = first_max_index - max_index
				second_difference = second_max_index - max_index
				if(first_difference >0 and second_difference>0 and first_difference>second_difference):
					max_sim = second_max
					mapping = second_mapping
				if(first_difference<0 and second_difference>0):
					max_sim = second_max
					mapping = second_mapping
				if(first_difference<0 and second_difference<0 and first_difference<second_difference):
					max_sim = second_max
					mapping = second_mapping
		print (f[0],mapping,max_sim)
		output += [(f[0],mapping,max_sim)]
		max_index = int(mapping.split('slide')[1].split('.jpg')[0])
		prev_mapping = mapping
		prev_max = max_sim


	#write output
	target = open(data_dir+'frame_mappings.txt','w')
	for o in output:
		target.write(str(o)+'\n')
	target.flush()
	target.close()

def compare(frames,slides, threshold):
	output = []
	l = len(frames)
	for i in range(3):
		output += [(frames[i][0],'slide0.jpg',1)]
	number = 0
	for i in range(2,l-1):
		s1 = 1 - spatial.distance.cosine(frames[i][1],frames[i+1][1])
		#print (frames[i][0],frames[i+1][0],s1)
		if(s1>threshold):
			print ((frames[i+1][0],'slide'+str(number)+'.jpg',1))
			output+=[(frames[i+1][0],'slide'+str(number)+'.jpg',1)]
		else:
			if(i+2<=l-1):
				s2 = 1 - spatial.distance.cosine(frames[i][1],frames[i+2][1])
			else:
				s2 = s1

			print ("s1",s1,"s2",s2)
			if (s2<=threshold):
				if(number<(len(slides)-1)):
					sim_next_slide = 1 - spatial.distance.cosine(frames[i+1][1],slides[number+1][1])
					print (number+1)
					print (sim_next_slide)
					sim_current_slide = 1 - spatial.distance.cosine(frames[i+1][1],slides[number][1])
					print (number)
					print (sim_current_slide)
					if(sim_next_slide > sim_current_slide and sim_next_slide>0.5):
						number = number + 1
						print ((frames[i+1][0],'slide'+str(number)+'.jpg',1))
						output += [(frames[i+1][0],'slide'+str(number)+'.jpg',1)]
					else:
						print ((frames[i+1][0],'slide'+str(number)+'.jpg',1))
						output += [(frames[i+1][0],'slide'+str(number)+'.jpg',1)]
				else:
					print ((frames[i+1][0],'slide'+str(number)+'.jpg',1))
					output += [(frames[i+1][0],'slide'+str(number)+'.jpg',1)]
			else:
				print ((frames[i+1][0],'slide'+str(number)+'.jpg',1))
				output += [(frames[i+1][0],'slide'+str(number)+'.jpg',1)]
	return (output,number)

def compare2(frames,slides, threshold):
	output = []
	l = len(frames)
	for i in range(3):
		output += [(frames[i][0],'slide0.jpg',1)]
	number = 0
	change = 0
	for i in range(2,l-1):
		s1 = 1 - spatial.distance.cosine(frames[i][1],frames[i+1][1])
		#print (frames[i][0],frames[i+1][0],s1)
		if(s1>threshold):
			print ((frames[i+1][0],'slide'+str(number)+'.jpg',1))
			output+=[(frames[i+1][0],'slide'+str(number)+'.jpg',1)]
		else:
			if(i+2<=l-1):
				s2 = 1 - spatial.distance.cosine(frames[i][1],frames[i+2][1])
			else:
				s2 = s1

			print ("s1",s1,"s2",s2)
			if (s2<=threshold):
				if(number<(len(slides)-1)):
					if(change!=i-1):
						
						#if(random.random()>0.3):
						#adding more logic unessarily
						max_sim = 1 - spatial.distance.cosine(frames[i+1][1],slides[number][1])
						max_sim = max_sim - (0.05)
						mapping = slides[number][0]
						print (mapping,max_sim)
						for s in slides[(number+1):(number+3)]:
							sim = 1 - spatial.distance.cosine(frames[i+1][1],s[1])
							print (s[0],sim)
							if(sim > max_sim):
								max_sim = sim
								mapping = s[0]
						if(max_sim>0.5):
							number = int(mapping.split('slide')[1].split('.jpg')[0])
							change = i
						'''
						else:
						sim_next_slide = 1 - spatial.distance.cosine(frames[i+1][1],slides[number+1][1])
						print (number+1)
						print (sim_next_slide)
						if(number < len(slides)-2):
							sim_next_next_slide = 1 - spatial.distance.cosine(frames[i+1][1],slides[number+2][1])
							print (number+2)
							print (sim_next_next_slide)
						else:
							sim_next_next_slide = 0
						if(sim_next_next_slide > sim_next_slide and sim_next_next_slide>0.5):
							number = number+2
							change = i
						else:
							if(sim_next_slide>sim_next_next_slide and sim_next_slide>0.5):
								number = number+1
								change = i
						'''
						print ((frames[i+1][0],'slide'+str(number)+'.jpg',1))
						output += [(frames[i+1][0],'slide'+str(number)+'.jpg',1)]
					else:
						print ((frames[i+1][0],'slide'+str(number)+'.jpg',1))
						output += [(frames[i+1][0],'slide'+str(number)+'.jpg',1)]
				else:
					print ((frames[i+1][0],'slide'+str(number)+'.jpg',1))
					output += [(frames[i+1][0],'slide'+str(number)+'.jpg',1)]
			else:
				print ((frames[i+1][0],'slide'+str(number)+'.jpg',1))
				output += [(frames[i+1][0],'slide'+str(number)+'.jpg',1)]
	return (output,number)


#new try -should have done before:
def map_frame_to_slide2(data_dir):
	fail = 0 
	slides_features = np.load(data_dir+'slides_features.npy').tolist()
	frame_features = np.load(data_dir+'frames_features.npy').tolist()
	slide_len = len(slides_features)
	print len(slides_features)
	print(len(slides_features[0]))
	frame_len =len(frame_features)
	print len(frame_features)
	print len(frame_features[0])
	new_framenames = []
	for i in range(frame_len):
		new_framenames +=['frame'+str(i)+'.jpg']
	framenames = new_framenames
	new_slidenames = []
	for i in range(slide_len):
		new_slidenames +=['slide'+str(i)+'.jpg']
	slidenames = new_slidenames
	print (slidenames)
	slides = zip(slidenames, slides_features)
	frames = zip(framenames, frame_features)
	(output,number) = compare(frames,slides,0.95)
	if(number != (len(slides)-1)):
		(output,number) = compare(frames,slides,0.97)
	if(number != (len(slides)-1)):
		(output,number) = compare(frames,slides,0.98)
	if(number != (len(slides)-1)):
		(output,number) = compare(frames,slides,0.99)
	if(number != (len(slides)-1)):
		(output,number) = compare(frames,slides,0.999)
	if(number!= (len(slides)-1)):
		#very complicated
		fail = 1

	#write output
	target = open(data_dir+'frame_mappings_new.txt','w')
	for o in output:
		target.write(str(o)+'\n')
	target.flush()
	target.close()

	return fail

def map_frame_to_slide3(data_dir):
	fail = 0 
	slides_features = np.load(data_dir+'slides_features.npy').tolist()
	frame_features = np.load(data_dir+'frames_features.npy').tolist()
	#slides_features.remove(slides_features[10])
	slide_len = len(slides_features)
	print len(slides_features)
	print(len(slides_features[0]))
	frame_len =len(frame_features)
	print len(frame_features)
	print len(frame_features[0])
	new_framenames = []
	for i in range(frame_len):
		new_framenames +=['frame'+str(i)+'.jpg']
	framenames = new_framenames
	new_slidenames = []
	for i in range(slide_len):
		new_slidenames +=['slide'+str(i)+'.jpg']
	slidenames = new_slidenames
	print (slidenames)
	slides = zip(slidenames, slides_features)
	frames = zip(framenames, frame_features)
	'''
	(output,number) = compare(frames,slides,0.95)
	if(number != (len(slides)-1)):
		(output,number) = compare(frames,slides,0.97)
	if(number != (len(slides)-1)):
		(output,number) = compare(frames,slides,0.98)
	if(number != (len(slides)-1)):
		(output,number) = compare(frames,slides,0.99)
	if(number != (len(slides)-1)):
		(output,number) = compare(frames,slides,0.999)
	if(number!= (len(slides)-1)):
	'''
	(output,number) = compare2(frames,slides,0.99)
	if(number!= (len(slides)-1)):
		fail = 1

	#write output
	target = open(data_dir+'frame_mappings_new.txt','w')
	for o in output:
		target.write(str(o)+'\n')
	target.flush()
	target.close()

	return fail








#check with one first and then move on
courses = ['data/cluster-analysis/']
failed_mappings = open(courses[0]+'/failed_mappings.txt','r')
failed_mappings =literal_eval(failed_mappings.read())

lectures = failed_mappings
fail_outputs = []
for data_dir in lectures:
	print (data_dir)
	contents = os.listdir(data_dir)
	if(('slides_features.npy' in contents) and ('frames_features.npy' in contents)):
		fail = map_frame_to_slide3(data_dir)
		if(fail):
			fail_outputs += [data_dir]
target = open(courses[0] + 'failed_mappings_2.txt','w')
target.write(str(fail_outputs)+'\n')
target.flush()
target.close()






