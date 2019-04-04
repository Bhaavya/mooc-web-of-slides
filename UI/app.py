import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask

app = Flask(__name__, instance_relative_config=True)
app.config.from_object('config')
application = app

from flask import render_template
from flask import request 
import urllib
import model


COURSE_NAMES = None
NUM_COURSES = None

@app.route('/')
def index():
	global COURSE_NAMES,NUM_COURSES
	COURSE_NAMES,NUM_COURSES = model.get_course_names()
	model.load_related_slides()
	return render_template("base.html",course_names=COURSE_NAMES,num_courses=NUM_COURSES)

@app.route('/webofslides/slide/<course_name>/<lno>')
def slide(course_name,lno):
	global COURSE_NAMES,NUM_COURSES
	if COURSE_NAMES is None and NUM_COURSES is None:
		COURSE_NAMES,NUM_COURSES = model.get_course_names()
	next_slide_name,lno,lec_name,(num_related_slides,related_slides,disp_str,related_course_names,rel_lnos,disp_color,disp_snippet),lec_names,lnos=model.get_next_slide(course_name,lno)
	return render_template("slide.html",slide_name=next_slide_name,course_name=course_name,num_related_slides=num_related_slides,related_slides = related_slides,disp_str=disp_str,disp_color=disp_color,disp_snippet=disp_snippet,related_course_names=related_course_names,lno=lno,lec_name=lec_name,lec_names=lec_names,lnos=lnos,course_names=COURSE_NAMES,num_courses=NUM_COURSES,rel_lnos=rel_lnos)


@app.route('/webofslides/related_slide/<course_name>/<lno>/<slide_name>')
def related_slide(course_name,slide_name,lno):
	global COURSE_NAMES,NUM_COURSES
	if COURSE_NAMES is None and NUM_COURSES is None:
		COURSE_NAMES,NUM_COURSES = model.get_course_names()
	next_slide_name,lno,lec_name,(num_related_slides,related_slides,disp_str,related_course_names,rel_lnos,disp_color,disp_snippet),lec_names,lnos=model.get_slide(course_name,slide_name,lno)
	return render_template("slide.html",slide_name=next_slide_name,course_name=course_name,num_related_slides=num_related_slides,related_slides = related_slides,disp_str=disp_str,disp_color=disp_color,disp_snippet=disp_snippet,related_course_names=related_course_names,lno=lno,lec_name=lec_name,lec_names=lec_names,lnos=lnos,course_names=COURSE_NAMES,num_courses=NUM_COURSES,rel_lnos=rel_lnos)

@app.route('/webofslides/next_slide/<course_name>/<lno>/<curr_slide>')
def next_slide(course_name,lno,curr_slide):
	global COURSE_NAMES,NUM_COURSES
	if COURSE_NAMES is None and NUM_COURSES is None:
		COURSE_NAMES,NUM_COURSES = model.get_course_names()
	next_slide_name,lno,lec_name,(num_related_slides,related_slides,disp_str,related_course_names,rel_lnos,disp_color,disp_snippet),lec_names,lnos =model.get_next_slide(course_name,lno,curr_slide)
	if next_slide_name is not None:
		return render_template("slide.html",slide_name=next_slide_name,course_name=course_name,num_related_slides=num_related_slides,related_slides = related_slides,disp_str=disp_str,disp_color=disp_color,disp_snippet=disp_snippet,related_course_names=related_course_names,lno=lno,lec_name=lec_name,lec_names=lec_names,lnos=lnos,course_names=COURSE_NAMES,num_courses=NUM_COURSES,rel_lnos=rel_lnos)
	else:
		return render_template("end.html",course_names=COURSE_NAMES,num_courses=NUM_COURSES)

@app.route('/webofslides/prev_slide/<course_name>/<lno>/<curr_slide>')
def prev_slide(course_name,lno,curr_slide):
	global COURSE_NAMES,NUM_COURSES
	if COURSE_NAMES is None and NUM_COURSES is None:
		COURSE_NAMES,NUM_COURSES = model.get_course_names()
	prev_slide_name,lno,lec_name,(num_related_slides,related_slides,disp_str,related_course_names,rel_lnos,disp_color,disp_snippet),lec_names,lnos=model.get_prev_slide(course_name,lno,curr_slide)
	if prev_slide_name is not None:
		return render_template("slide.html",slide_name=prev_slide_name,course_name=course_name,num_related_slides=num_related_slides,related_slides = related_slides,disp_str=disp_str,disp_color=disp_color,disp_snippet=disp_snippet,related_course_names=related_course_names,lno=lno,lec_name=lec_name,lec_names=lec_names,lnos=lnos,course_names=COURSE_NAMES,num_courses=NUM_COURSES,rel_lnos=rel_lnos)
	else:
		return render_template("end.html",course_names=COURSE_NAMES,num_courses=NUM_COURSES)


@app.route('/webofslides/end')
def end():
	global COURSE_NAMES,NUM_COURSES
	if COURSE_NAMES is None and NUM_COURSES is None:
		COURSE_NAMES,NUM_COURSES = model.get_course_names()
	return render_template("end.html",course_names=COURSE_NAMES,num_courses=NUM_COURSES)









if __name__ == '__main__':
	app.run()