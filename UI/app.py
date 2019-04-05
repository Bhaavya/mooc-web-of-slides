import sys
import os
import json 
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask

app = Flask(__name__, instance_relative_config=True)
app.config.from_object('config')
application = app

from flask import render_template
from flask import request,jsonify
import urllib
import model
import forms
import datetime

COURSE_NAMES = None
NUM_COURSES = None

@app.route('/')
def index():
    global COURSE_NAMES,NUM_COURSES
    COURSE_NAMES,NUM_COURSES = model.get_course_names()
    model.load_related_slides()
    return render_template("base.html",course_names=COURSE_NAMES,num_courses=NUM_COURSES)

@app.route('/webofslides/slide/<course_name>/<lno>')
def slide(course_name,lno,render=True,action='drop-down'):
	global COURSE_NAMES,NUM_COURSES
	if COURSE_NAMES is None and NUM_COURSES is None:
		COURSE_NAMES,NUM_COURSES = model.get_course_names()
	next_slide_name,lno,lec_name,(num_related_slides,related_slides,disp_str,related_course_names,rel_lnos,disp_color,disp_snippet),lec_names,lnos=model.get_next_slide(course_name,lno)
	model.log(request.remote_addr,next_slide_name,datetime.datetime.now(),action)
	if render:
		return render_template("slide.html",slide_name=next_slide_name,course_name=course_name,num_related_slides=num_related_slides,related_slides = related_slides,disp_str=disp_str,disp_color=disp_color,disp_snippet=disp_snippet,related_course_names=related_course_names,lno=lno,lec_name=lec_name,lec_names=lec_names,lnos=lnos,course_names=COURSE_NAMES,num_courses=NUM_COURSES,rel_lnos=rel_lnos)


@app.route('/webofslides/related_slide/<course_name>/<lno>/<slide_name>')
def related_slide(course_name,slide_name,lno,render=True,action='related'):
	global COURSE_NAMES,NUM_COURSES
	if COURSE_NAMES is None and NUM_COURSES is None:
		COURSE_NAMES,NUM_COURSES = model.get_course_names()
	next_slide_name,lno,lec_name,(num_related_slides,related_slides,disp_str,related_course_names,rel_lnos,disp_color,disp_snippet),lec_names,lnos=model.get_slide(course_name,slide_name,lno)
	model.log(request.remote_addr,next_slide_name,datetime.datetime.now(),action)
	if render:
		return render_template("slide.html",slide_name=next_slide_name,course_name=course_name,num_related_slides=num_related_slides,related_slides = related_slides,disp_str=disp_str,disp_color=disp_color,disp_snippet=disp_snippet,related_course_names=related_course_names,lno=lno,lec_name=lec_name,lec_names=lec_names,lnos=lnos,course_names=COURSE_NAMES,num_courses=NUM_COURSES,rel_lnos=rel_lnos)

@app.route('/webofslides/search_slide/<course_name>/<lno>/<slide_name>/<idx>')
def search_slide(course_name,slide_name,lno,render=True,action='search_result'):
	related_slide(course_name,slide_name,lno,render=render,action='{}-{}'.format(action,idx))

@app.route('/webofslides/next_slide/<course_name>/<lno>/<curr_slide>')
def next_slide(course_name,lno,curr_slide,render=True,action='next'):
	global COURSE_NAMES,NUM_COURSES
	if COURSE_NAMES is None and NUM_COURSES is None:
		COURSE_NAMES,NUM_COURSES = model.get_course_names()
	next_slide_name,lno,lec_name,(num_related_slides,related_slides,disp_str,related_course_names,rel_lnos,disp_color,disp_snippet),lec_names,lnos =model.get_next_slide(course_name,lno,curr_slide)
	if next_slide_name is not None:
		model.log(request.remote_addr,next_slide_name,datetime.datetime.now(),action)
		if render:
			return render_template("slide.html",slide_name=next_slide_name,course_name=course_name,num_related_slides=num_related_slides,related_slides = related_slides,disp_str=disp_str,disp_color=disp_color,disp_snippet=disp_snippet,related_course_names=related_course_names,lno=lno,lec_name=lec_name,lec_names=lec_names,lnos=lnos,course_names=COURSE_NAMES,num_courses=NUM_COURSES,rel_lnos=rel_lnos)
	else:
		model.log(request.remote_addr,'End',datetime.datetime.now(),action)
		if render:
			return render_template("end.html",course_names=COURSE_NAMES,num_courses=NUM_COURSES)

@app.route('/webofslides/prev_slide/<course_name>/<lno>/<curr_slide>')
def prev_slide(course_name,lno,curr_slide,render=True,action='prev'):
	global COURSE_NAMES,NUM_COURSES
	if COURSE_NAMES is None and NUM_COURSES is None:
		COURSE_NAMES,NUM_COURSES = model.get_course_names()
	prev_slide_name,lno,lec_name,(num_related_slides,related_slides,disp_str,related_course_names,rel_lnos,disp_color,disp_snippet),lec_names,lnos=model.get_prev_slide(course_name,lno,curr_slide)
	if prev_slide_name is not None:
		model.log(request.remote_addr,prev_slide_name,datetime.datetime.now(),action)
		if render:
			return render_template("slide.html",slide_name=prev_slide_name,course_name=course_name,num_related_slides=num_related_slides,related_slides = related_slides,disp_str=disp_str,disp_color=disp_color,disp_snippet=disp_snippet,related_course_names=related_course_names,lno=lno,lec_name=lec_name,lec_names=lec_names,lnos=lnos,course_names=COURSE_NAMES,num_courses=NUM_COURSES,rel_lnos=rel_lnos)
	else:
		model.log(request.remote_addr,'End',datetime.datetime.now(),action)
		if render:
			return render_template("end.html",course_names=COURSE_NAMES,num_courses=NUM_COURSES)


@app.route('/webofslides/end')
def end():
    global COURSE_NAMES,NUM_COURSES
    if COURSE_NAMES is None and NUM_COURSES is None:
        COURSE_NAMES,NUM_COURSES = model.get_course_names()
    return render_template("end.html",course_names=COURSE_NAMES,num_courses=NUM_COURSES)

@app.route('/webofslides/prev_slide/<course_name>/<lno>/<curr_slide>', methods=['POST'])
@app.route('/webofslides/next_slide/<course_name>/<lno>/<curr_slide>', methods=['POST'])
@app.route('/webofslides/related_slide/<course_name>/<lno>/<slide_name>', methods=['POST'])
@app.route('/webofslides/slide/<course_name>/<lno>', methods=['POST'])
@app.route('/webofslides/end', methods=['POST'])
@app.route('/', methods=['POST'])
def results(course_name=None, lno=None, slide_name=None, curr_slide=None):
    search = forms.SearchForm(request.form)
    #if request.method == 'POST':
    return search_results(search)
 
@app.route('/results')
def search_results(search):
    results = []
    search_string = request.form['search']
    #if search.data['search'] == '':
    #    flash('No results found!')
    #    return redirect('/')
    model.log(request.remote_addr,search_string,datetime.datetime.now(),'search_query')
    num_results,results,disp_strs,search_course_names,lnos, snippets = model.get_search_results(search_string)
    if not results:
        return render_template("search.html",num_results=0,results = [],disp_strs=disp_strs,search_course_names=search_course_names,lnos=lnos,course_names=COURSE_NAMES,num_courses=NUM_COURSES)
    else:
        # display results
        return render_template("search.html",num_results=num_results,results = results,disp_strs=disp_strs,search_course_names=search_course_names,lnos=lnos,course_names=COURSE_NAMES,num_courses=NUM_COURSES, snippets=snippets)


@app.route('/webofslides/log_action',methods=['GET', 'POST'])
def log_action():
	request_dict = json.loads(request.data)
	action = request_dict['action']
	route = request_dict['route']
	print(action,route)
	if action is not None and route is not None:
		route_ele = route.split('/')
		func_type = route_ele[2]
		print (func_type,route_ele)
		if func_type == 'related_slide':
			related_slide(route_ele[3],route_ele[5],route_ele[4],render=False,action=action)
		elif func_type == 'next_slide':
			next_slide(route_ele[3],route_ele[4],route_ele[5],render=False,action=action)
		elif func_type == 'prev_slide':
			prev_slide(route_ele[3],route_ele[4],route_ele[5],render=False,action=action)
		elif func_type == 'slide':
			slide(route_ele[3],route_ele[4],render=False,action=action)
		resp = jsonify(success=True)
	else:
		resp = jsonify(success=False)
	return resp 


if __name__ == '__main__':
    app.run()