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

def resolve_slide(course_name,lno,type_,slide_name=None,log=False,action=None):
	global COURSE_NAMES,NUM_COURSES
	if COURSE_NAMES is None and NUM_COURSES is None:
		COURSE_NAMES,NUM_COURSES = model.get_course_names()
	if type_ =='drop-down':
		ret = model.get_next_slide(course_name,lno)
	elif type_ == 'related':
		ret = model.get_slide(course_name,slide_name,lno)
	elif type_ == 'next':
		ret = model.get_next_slide(course_name,lno,slide_name)
	elif type_ == 'prev':
		ret = model.get_prev_slide(course_name,lno,slide_name)
	if log:
		if ret[0] is not None:
			model.log(request.remote_addr,ret[0],datetime.datetime.now(),action)
		else:
			model.log(request.remote_addr,'End',datetime.datetime.now(),action)
	return ret 
	
@app.route('/webofslides/slide/<course_name>/<lno>')
def slide(course_name,lno):
	next_slide_name,lno,lec_name,(num_related_slides,related_slides,disp_str,related_course_names,rel_lnos,disp_color,disp_snippet),lec_names,lnos= resolve_slide(course_name,lno,'drop-down')
	return render_template("slide.html",slide_name=next_slide_name,course_name=course_name,num_related_slides=num_related_slides,related_slides = related_slides,disp_str=disp_str,disp_color=disp_color,disp_snippet=disp_snippet,related_course_names=related_course_names,lno=lno,lec_name=lec_name,lec_names=lec_names,lnos=lnos,course_names=COURSE_NAMES,num_courses=NUM_COURSES,rel_lnos=rel_lnos)


@app.route('/webofslides/related_slide/<course_name>/<lno>/<slide_name>')
def related_slide(course_name,slide_name,lno):
	next_slide_name,lno,lec_name,(num_related_slides,related_slides,disp_str,related_course_names,rel_lnos,disp_color,disp_snippet),lec_names,lnos=resolve_slide(course_name,lno,'related',slide_name=slide_name)
	print (next_slide_name,'+'*20)
	return render_template("slide.html",slide_name=next_slide_name,course_name=course_name,num_related_slides=num_related_slides,related_slides = related_slides,disp_str=disp_str,disp_color=disp_color,disp_snippet=disp_snippet,related_course_names=related_course_names,lno=lno,lec_name=lec_name,lec_names=lec_names,lnos=lnos,course_names=COURSE_NAMES,num_courses=NUM_COURSES,rel_lnos=rel_lnos)

@app.route('/webofslides/search_slide/<course_name>/<lno>/<slide_name>')
def search_slide(course_name,slide_name,lno):
	related_slide(course_name,slide_name,lno)

@app.route('/webofslides/next_slide/<course_name>/<lno>/<curr_slide>')
def next_slide(course_name,lno,curr_slide):
	next_slide_name,lno,lec_name,(num_related_slides,related_slides,disp_str,related_course_names,rel_lnos,disp_color,disp_snippet),lec_names,lnos = resolve_slide(course_name,lno,'next',slide_name=curr_slide)
	if next_slide_name is not None:
		return render_template("slide.html",slide_name=next_slide_name,course_name=course_name,num_related_slides=num_related_slides,related_slides = related_slides,disp_str=disp_str,disp_color=disp_color,disp_snippet=disp_snippet,related_course_names=related_course_names,lno=lno,lec_name=lec_name,lec_names=lec_names,lnos=lnos,course_names=COURSE_NAMES,num_courses=NUM_COURSES,rel_lnos=rel_lnos)
	else:
		return render_template("end.html",course_names=COURSE_NAMES,num_courses=NUM_COURSES)

@app.route('/webofslides/prev_slide/<course_name>/<lno>/<curr_slide>')
def prev_slide(course_name,lno,curr_slide):
	prev_slide_name,lno,lec_name,(num_related_slides,related_slides,disp_str,related_course_names,rel_lnos,disp_color,disp_snippet),lec_names,lnos=resolve_slide(course_name,lno,'prev',slide_name=curr_slide)
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

@app.route('/webofslides/prev_slide/<course_name>/<lno>/<curr_slide>', methods=['POST'])
@app.route('/webofslides/next_slide/<course_name>/<lno>/<curr_slide>', methods=['POST'])
@app.route('/webofslides/related_slide/<course_name>/<lno>/<slide_name>', methods=['POST'])
@app.route('/webofslides/slide/<course_name>/<lno>', methods=['POST'])
@app.route('/webofslides/end', methods=['POST'])
@app.route('/', methods=['POST'])
def results(course_name=None, lno=None, slide_name=None, curr_slide=None):
    search = forms.SearchForm(request.form)
    #if request.method == 'POST':
    model.log(request.remote_addr,request.form['search'],datetime.datetime.now(),'search_query')
    return search_results(search)
 
@app.route('/results')
def search_results(search):
    results = []
    search_string = request.form['search']
    #if search.data['search'] == '':
    #    flash('No results found!')
    #    return redirect('/')
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
			resolve_slide(route_ele[3],route_ele[4],'related',slide_name=route_ele[5],log=True,action=action)
		elif func_type == 'next_slide':
			resolve_slide(route_ele[3],route_ele[4],'next',slide_name=route_ele[5],log=True,action=action)
		elif func_type == 'prev_slide':
			resolve_slide(route_ele[3],route_ele[4],'prev',slide_name=route_ele[5],log=True,action=action)
		elif func_type == 'slide':
			resolve_slide(route_ele[3],route_ele[4],'drop-down',log=True,action=action)
		elif func_type == 'search_slide':
			resolve_slide(route_ele[3],route_ele[4],'search_results',slide_name=route_ele[5],log=True,action=action)
		resp = jsonify(success=True)
	else:
		resp = jsonify(success=False)
	return resp 


if __name__ == '__main__':
    app.run()