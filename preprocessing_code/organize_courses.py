import os 
import difflib
import shutil 
'''
Organize lessons from all courses in root_dir and save in new_folder
'''

def organize_courses(root_dir,new_folder):
	courses = {}
	lessons = {}
	f=0
	#Parse all courses in root_dir
	for subdir, dirs, files in os.walk(root_dir):
		#new course folder
		if os.path.dirname(subdir) == root_dir:	
			if f!=0:
				courses[course_name] = lessons
				lessons = {}
			else:
				f=1
			course_name = os.path.basename(subdir)
			print ("Processing:\t",course_name)
			course_dir = subdir
		#a subdirectory of course
		elif subdir != root_dir:
			other_files = []
			rel_vids = []
			for file_name in files:
				path_name = os.path.join(subdir,file_name)
				#only need video and corresponding pdf and subtitle
				if file_name.lower().endswith('.mp4'):
					rel_vids.append(path_name)
				elif file_name.lower().endswith(('.en.srt', '.pdf')):
					other_files.append(path_name)
			lessons.update(form_lessons(other_files,rel_vids))
	courses[course_name] = lessons
	save_courses(courses,new_folder)

def form_lessons(other_files,rel_vids):
	#match pdf/subtitles to lesson videos
	lessons = {}
	for vid in rel_vids:
		lessons[vid] = []

	for file_name in other_files:
		matching_vid = difflib.get_close_matches(file_name,rel_vids,1,0.2)
		if len(matching_vid) == 1:
			lessons[matching_vid[0]].append(file_name)
		else:
			pass
			# print ("*********No matching video found :*************",file_name,rel_vids)
	return lessons

def save_courses(courses,new_folder):
	#copy organized lessons in new directory
	for course,lessons in courses.items():
		print ("Saving:",course)
		course_dir = os.path.join(new_folder,course)
		os.mkdir(course_dir)
		lno = 0
		#make a new folder for each lesson and copy the 3 lesson files
		for lesson,files in lessons.items():
			lno+=1
			lesson_dir = os.path.join(course_dir,str(lno))
			os.mkdir(lesson_dir)
			#copied files are named with the whole path after course name including any subfolder names. folder names are separated by '##'
			old_name = os.path.join(lesson_dir,os.path.basename(lesson))
			new_name = os.path.join(lesson_dir, lesson.split(course)[1][1:].replace(os.path.sep,'##'))
			shutil.copy(lesson, lesson_dir)
			os.rename(old_name,new_name)
			for file in files:
				old_name = os.path.join(lesson_dir,os.path.basename(file))
				new_name = os.path.join(lesson_dir, file.split(course)[1][1:].replace(os.path.sep,'##'))
				shutil.copy(file, lesson_dir)
				os.rename(old_name,new_name)
		
def main():
	organize_courses('/Users/bhavya/Documents/GitHub/mooc-web-of-slides/high_res_data','/Users/bhavya/Documents/GitHub/mooc-web-of-slides/test')

if __name__ == '__main__':
	main()