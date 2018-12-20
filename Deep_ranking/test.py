failures_names = open('logreg_results/Logistic_regression_8_failures_names.txt' , 'r').readlines()
all_names = open('../data/slide_names_for_training.txt','r').readlines()
last_slides = 0
first_slides = 0
for i in range(384):
	next_slide = failures_names[i].split('##')[-1]
	slide_number = int(next_slide.strip('\n').split('slide')[1])
 	print slide_number
	next_name = '##'.join(failures_names[i].split('##')[:-1])+'##slide'+str(slide_number+1)+'\n'
	before_name = '##'.join(failures_names[i].split('##')[:-1])+'##slide'+str(slide_number-1)+'\n'
	print next_name
	print before_name
	if next_name not in all_names:
		last_slides +=1
		#print failures_names[i]
	if before_name not in all_names:
		first_slides +=1
		#print failures_names[i]
print last_slides
print first_slides
