import numpy as np
from PIL import Image
import os
from scipy import spatial
import indicoio
import time
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

def get_indicoio_features(data_dir, local_path):
	if(local_path == 'slides'):
		pathin = data_dir+'new'+local_path+'/'
	else:
		pathin = data_dir+local_path+'/'
	names = os.listdir(pathin)
	l = len(names)
	new_names = []
	for i in range(l):
		new_names +=[local_path[:-1]+str(i)+'.jpg']
	names = new_names
	
		
	paths = []
	i=0
	for f in names:
		i = i+1
		paths += [pathin+f]
		if(i%100) ==0:
			print(i)

	features = []
	for i in range(0,len(paths),20):
		features +=  indicoio.image_features(paths[i:(i+20)])
		print ((i+20), " done")
	print ("reading "+str(local_path)+" done")
	#write to files
	pathout = data_dir+local_path+'_features.npy'
	np.save(pathout,np.asarray(features))



def map_frame_to_slide(data_dir):
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
		max_sim = 0
		mapping = 'slide0.jpg'
		for s in slides[max((max_index-2),0):]:
			s1 = 1 - spatial.distance.cosine(f[1],s[1])
			print (s[0],s1)
			if(s1 > max_sim):
				max_sim = s1
				mapping = s[0]
		if(max_sim<0.50):
			mapping = prev_mapping
			max_sim = prev_max
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

def map_frame_to_slide_short(data_dir):
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
	prev_mapping = 'slide0.jpg'
	prev_max = 0
	max_index = 0
	#map each frame to a slide:
	output = []
	for f in frames:
		max_sim = 0
		mapping = 'slide0.jpg'
		for s in slides[max((max_index-2),0):(max_index+3)]:
			s1 = 1 - spatial.distance.cosine(f[1],s[1])
			print (s[0],s1)
			if(s1 > max_sim):
				max_sim = s1
				mapping = s[0]
		if(max_sim<0.50):
			mapping = prev_mapping
			max_sim = prev_max
		print (f[0],mapping,max_sim)
		output += [(f[0],mapping,max_sim)]
		max_index = int(mapping.split('slide')[1].split('.jpg')[0])
		prev_mapping = mapping
		prev_max = max_sim


	#write output
	target = open(data_dir+'frame_mappings_test.txt','w')
	for o in output:
		target.write(str(o)+'\n')
	target.flush()
	target.close()


#check with one first and then move on

#toremove = ['data/language-processing/32', 'data/language-processing/35', 'data/language-processing/34', 'data/language-processing/33', 'data/language-processing/20', 'data/language-processing/18', 'data/language-processing/27', 'data/language-processing/9', 'data/language-processing/11', 'data/language-processing/7', 'data/language-processing/29', 'data/language-processing/16', 'data/language-processing/42', 'data/language-processing/6', 'data/language-processing/28', 'data/language-processing/17', 'data/language-processing/1', 'data/language-processing/10', 'data/language-processing/19', 'data/language-processing/26']
#for data_dir in toremove:
#	print (data_dir)
#map_frame_to_slide(data_dir+'/')

'''
courses = ['data/language-processing/']
toremove = ['data/language-processing/5','data/language-processing/32', 'data/language-processing/35', 'data/language-processing/34', 'data/language-processing/33', 'data/language-processing/20', 'data/language-processing/18', 'data/language-processing/27', 'data/language-processing/9', 'data/language-processing/11', 'data/language-processing/7', 'data/language-processing/29', 'data/language-processing/16', 'data/language-processing/42', 'data/language-processing/6', 'data/language-processing/28', 'data/language-processing/17', 'data/language-processing/1', 'data/language-processing/10', 'data/language-processing/19', 'data/language-processing/26']
lectures = []
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
for data_dir in lectures:
	print (data_dir)
	map_frame_to_slide(data_dir+'/')
'''

courses = ['data/language-processing/','data/recommender-systems-introduction/','data/cluster-analysis/']
courses = ['data/cs-410/']
toremove = ['data/language-processing/32', 'data/language-processing/35', 'data/language-processing/34', 'data/language-processing/33', 'data/language-processing/20', 'data/language-processing/18', 'data/language-processing/27', 'data/language-processing/9', 'data/language-processing/11', 'data/language-processing/7', 'data/language-processing/29', 'data/language-processing/16', 'data/language-processing/42', 'data/language-processing/6', 'data/language-processing/28', 'data/language-processing/17', 'data/language-processing/1', 'data/language-processing/10', 'data/language-processing/19', 'data/language-processing/26']
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
  index = lectures.index('data/cs-410/82')
  print (lectures[index:])

  for l in lectures[index:]:
    pdf_inpath = ''
    video_inpath = ''
    contents = os.listdir(l)
    if ('newslides' in contents):
      print (l+'/'+'newslides/')
      get_indicoio_features(l+'/','slides')
    if ('frames' in  contents):
      print (l+'/'+'frames/')
      get_indicoio_features(l+'/','frames')



