import os
if not os.path.exists('data/slides_augmented_content'):
    os.mkdir('data/slides_augmented_content')
data_dir = 'data/slides_augmented_content/'
  
courses = ['data/cs-410/','data/cluster-analysis/','data/language-processing/','data/bayesian_methods_in_machine_learning/', 'data/recommender-systems-introduction/']


for c in courses:
  #root, dirs, files = os.walk(c).next()
  root, dirs, files = next(os.walk(c))
  lectures_names = dirs
  if c in lectures_names:
    lectures_names.remove(c)
  lectures = [c+l for l in lectures_names]
  coursepath = data_dir + c[5:]
  if not os.path.exists(coursepath):
  	os.mkdir(coursepath)
  i=0
  for l in lectures:
  	lecture_name = lectures_names[i]
  	i = i+1
	contents = os.listdir(l)
	if('slide_augmented_content.txt' in contents):
		print (l)
		for s in contents:
			if '.pdf' in s:
				slide_title = s
		slide_title = slide_title[:-4]
		if not os.path.exists(coursepath+lecture_name):
			os.mkdir(coursepath+lecture_name)
		with open(l+'/'+'slide_augmented_content.txt') as f:
			with open(coursepath+lecture_name+'/'+slide_title +'.txt', "w") as f1:
				for line in f:
					f1.write(line)


   



