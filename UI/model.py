import os
import re 
import metapy

static_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'static')
slides_path = os.path.join(static_path,'slides')
related_slides_path = os.path.join(static_path,'ranking_results.csv')
related_dict = {}

idx = metapy.index.make_inverted_index('slides-config.toml')
ranker = metapy.index.OkapiBM25()
slide_titles = []
with open(os.path.join('./slides/slides.dat.labels')) as f:
    for line in f:
        slide_titles.append(line[:-1])

def get_course_names():
    course_names = sorted(os.listdir(slides_path))
    num_course = len(course_names)
    return course_names,num_course

def load_related_slides():
    global related_dict
    with open(related_slides_path,'r') as f:
        related_slides = f.readlines()
    for row in related_slides:
        cols = row.split(',')
        key = cols[0].replace('##','----')+'.pdf'
        related_dict[key] = []
        for col_num in range(1,len(cols),2):
            pdf_name = cols[col_num].replace('##','----')+'.pdf'
            name_comp = pdf_name.split('----')
            course_name = name_comp[0]
            lec_name ='----'.join(name_comp[1:-1])
            if os.path.exists(os.path.join(slides_path,course_name,lec_name,pdf_name)):
                related_dict[key].append(pdf_name)

def sort_slide_names(l): 
    """ Sort the given iterable in the way that humans expect.""" 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def get_slide(course_name,slide,lno):
    lectures = sort_slide_names(os.listdir(os.path.join(slides_path, course_name)))
    lno = int(lno)
    related_slides_info = get_related_slides(slide)
    return slide,lno,lectures[lno],related_slides_info,lectures,range(len(lectures))


def get_next_slide(course_name,lno,curr_slide=None):
    lectures = sort_slide_names(os.listdir(os.path.join(slides_path, course_name)))
    lno = int(lno)
    slides = sort_slide_names(os.listdir(os.path.join(slides_path,course_name,lectures[lno])))
    if curr_slide is not None:
        idx = slides.index(curr_slide)
        slides = slides[idx+1:]
    if len(slides)>0:
        next_slide = slides[0]
    else:
        if lno==len(lectures)-1:
            return None,None,None,(None,None,None,None,None),None,None
        else:
            next_slide = sort_slide_names(os.listdir(os.path.join(slides_path,course_name,lectures[lno+1])))[0]
            lno+=1
    related_slides_info = get_related_slides(next_slide)
    return next_slide, lno,lectures[lno],related_slides_info,lectures,range(len(lectures))    

def get_prev_slide(course_name,lno,curr_slide):
    lectures = sort_slide_names(os.listdir(os.path.join(slides_path, course_name)))
    lno = int(lno)
    slides = sort_slide_names(os.listdir(os.path.join(slides_path,course_name,lectures[lno])))
    idx = slides.index(curr_slide)
    if idx==0:
            if lno==0:
                return None,None,None,(None,None,None,None,None),None,None
            else:
                prev_slide = sort_slide_names(os.listdir(os.path.join(slides_path,course_name,lectures[lno-1])))[-1]
                lno-=1
    else:
        prev_slide = slides[:idx][-1]
    related_slides_info = get_related_slides(prev_slide)
    return prev_slide ,lno,lectures[lno],related_slides_info,lectures,range(len(lectures))    

def get_related_slides(slide_name):
    if related_dict=={}:
        load_related_slides()
    related_slides = related_dict[slide_name]
    disp_strs = []
    course_names = []
    lnos = []
    for r in related_slides:
        comp = r.split('----')
        disp_strs.append(' '.join(comp[0].replace('_','-').split('-')).title() + ' : ' + ' '.join(comp[-2].replace('.txt','').replace('_','-').split('-')).title() + ' , ' + ' '.join(comp[-1].replace('.pdf','').split('-')).title())
        course_names.append(comp[0])
        lectures = sort_slide_names(os.listdir(os.path.join(slides_path, comp[0])))
        lnos.append(lectures.index('----'.join(comp[1:-1])))
    return len(related_slides),related_slides,disp_strs,course_names,lnos

def load_search_results(search):
    query = metapy.index.Document()
    query.content(search)
    top_docs = ranker.score(idx, query, num_results=10)
    results = [slide_titles[x[0]] for x in top_docs]
    disp_strs = []
    course_names = []
    lnos = []
    for r in results:
        comp = r.split('##')
        disp_strs.append(' '.join(comp[0].replace('_','-').split('-')).title() + ' : ' + ' '.join(comp[-2].replace('.txt','').replace('_','-').split('-')).title() + ' , ' + ' '.join(comp[-1].replace('.pdf','').split('-')).title())
        course_names.append(comp[0])
        lectures = sort_slide_names(os.listdir(os.path.join(slides_path, comp[0])))
        lnos.append(lectures.index('----'.join(comp[1:-1])))
    return len(results),results,disp_strs,course_names,lnos

def get_search_results(course_name,lno,curr_slide):
    lectures = sort_slide_names(os.listdir(os.path.join(slides_path, course_name)))
    lno = int(lno)
    slides = sort_slide_names(os.listdir(os.path.join(slides_path,course_name,lectures[lno])))
    idx = slides.index(curr_slide)
    related_slides_info = get_search_results(search)
    return curr_slide, lno,lectures[lno],related_slides_info,lectures,range(len(lectures))    


if __name__ == '__main__':
    load_related_slides()
    print (related_dict)

