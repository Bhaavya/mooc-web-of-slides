import os
import re 
import io
import numpy as np
import pickle
import metapy

static_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'static')
slides_path = os.path.join(static_path,'slides')
related_slides_path = os.path.join(static_path,'ranking_results.csv')
related_dict = {}
slide_names = open(os.path.join(static_path,'slide_names2.txt'), 'r').readlines()
slide_names = [name.strip() for name in slide_names]
slide_titles = io.open(os.path.join(static_path,'slide_titles.txt'), 'r', encoding='utf-8').readlines()
slide_titles = [t.strip() for t in slide_titles]
title_mapping = dict(zip(slide_names, slide_titles))

vocabulary_list = pickle.load(open('static/tf_idf_outputs/vocabulary_list.p', 'rb'))
tfidfs = np.load("static/tf_idf_outputs/normalized_tfidfs.npy")
title_tfidfs = np.load("static/tf_idf_outputs/normalized_title_tfidfs.npy")
ss_corpus = pickle.load(open('static/tf_idf_outputs/ss_corpus.p', 'rb'))

log_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'log','log.txt')

def log(ip,to_slide,action,start_time):
    with open(log_path,'a+') as f:
        f.write('{},{},{},{}\n'.format(ip,to_slide,action,start_time))
        
def get_snippet_sentences(slide_name, matching_keywords):
    idx = slide_names.index(slide_name)
    content = ss_corpus[idx].split(' ')
    include  = [0]*len(content)
    for c in range(len(content)):
        if content[c] in matching_keywords:
            for i in range(max(0,c-2), min(c+3,len(content))):
                include[i] = 1
    text = '' 
    for c in range(len(content)):
        if include[c]== 1:
            if c!=0 and include[c-1] == 0:
                text += '......'
            text += content[c] + ' '
    text += '......'
    return text

def trim_name(slide_name):
    name = slide_name.split(' ')
    new_name = []
    for n in name:
        if (len(n) > 2) and (len(re.findall('[0-9\.]+', n)) == 0):
            if (n == 'Lesson') or (n=='Part'):
                continue
            new_name += [n]

    return ' '.join(new_name)

def get_color(slide_course_name, related_slide_course_name):
    if slide_course_name==related_slide_course_name:
        return "blue"
    else:
        return "brown"

def get_snippet(slide_name, related_slide_name):
    no_keywords = False
    related_slide_name = related_slide_name.replace('----', '##')[:-4]
    slide_name = slide_name.replace('----', '##')[:-4]
    idx1 = slide_names.index(slide_name)
    idx2 = slide_names.index(related_slide_name)    
    title_tfidf1 = title_tfidfs[idx1,:]
    title_tfidf2 = title_tfidfs[idx2,:]
    tfidf1 = tfidfs[idx1,:]
    tfidf2 = tfidfs[idx2,:]
    term_sims = 2.8956628*(title_tfidf1*title_tfidf2)+ 5.92724651*(tfidf1*tfidf2)
    top_terms_indeces = np.argsort(term_sims)[::-1][:5]
    #print related_slide_name
    #print np.sort(term_sims)[::-1][:10]
    top_terms_indeces = filter(lambda l : term_sims[l]>0, top_terms_indeces)
    matching_words = [vocabulary_list[t] for t in top_terms_indeces]
    #matching_words = [(vocabulary_list[t],vec.idf_[t]) for t in top_terms_indeces]
    #matching_words = sorted(matching_words, key = lambda l :l[1], reverse = True)
    #matching_words = map(lambda l : l[0], matching_words)
    if len(matching_words) == 0 :
        no_keywords = True
    keywords = ', '.join(matching_words) 
    snippet_sentence =  get_snippet_sentences(related_slide_name, matching_words)

    return (('Slide title : ' + title_mapping[related_slide_name][:-1] +'\n' + 'Matching keywords: ' + keywords + '\n' + 'Snippet:' + snippet_sentence),no_keywords)

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
            if cols[col_num+1].strip() != '':
                score = float(cols[col_num+1].strip())
                if score < 0.03:
                    break
            name_comp = pdf_name.split('----')
            course_name = name_comp[0]
            lec_name ='----'.join(name_comp[1:-1])
            if os.path.exists(os.path.join(slides_path,course_name,lec_name,pdf_name)):
                related_dict[key].append(pdf_name)

def sort_slide_names(l): 
    """ Sort the given iterable in the way that humans expect.""" 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key) ] 
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
            return None,None,None,(None,None,None,None,None,None,None),None,None
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
                return None,None,None,(None,None,None,None,None,None,None),None,None
            else:
                prev_slide = sort_slide_names(os.listdir(os.path.join(slides_path,course_name,lectures[lno-1])))[-1]
                lno-=1
    else:
        prev_slide = slides[:idx][-1]
    related_slides_info = get_related_slides(prev_slide)
    return prev_slide, lno,lectures[lno],related_slides_info,lectures,range(len(lectures))    


def get_related_slides(slide_name):
    if related_dict=={}:
        load_related_slides()
    filtered_related_slides = []
    disp_strs = []
    disp_colors = []
    disp_snippets= []
    course_names = []
    lnos = []
    slide_comp = slide_name.split('----')
    related_slide_trim_names = []
    if slide_name in related_dict:
        related_slides = related_dict[slide_name]
        filtered_related_slides = []
        for r in related_slides:
            comp = r.split('----')
            #disp_strs.append(' '.join(comp[0].replace('_','-').split('-')).title() + ' : ' + ' '.join(comp[-2].replace('.txt','').replace('_','-').split('-')).title() + ' , ' + ' '.join(comp[-1].replace('.pdf','').split('-')).title())
            related_slide_name = ' '.join(comp[-2].replace('.txt','').replace('_','-').split('-')).title() 
            slide_course_name = ' '.join(slide_comp[0].replace('_','-').split('-')).title()
            related_slide_course_name = ' '.join(comp[0].replace('_','-').split('-')).title()
            trimmed_name = ' '.join(comp[0].replace('_','-').split('-')).title() + ' : ' + trim_name(related_slide_name)
            if trimmed_name in related_slide_trim_names:
                continue
            else:
                related_slide_trim_names += [trimmed_name]
            color = get_color(slide_course_name, related_slide_course_name)
            snippet,no_keywords = get_snippet(slide_name, r)
            if no_keywords==True:
                continue
            filtered_related_slides.append(r)
            disp_strs.append(' '.join(comp[0].replace('_','-').split('-')).title() + ' : ' + trim_name(related_slide_name))
            disp_snippets.append(snippet)
            disp_colors.append(color)
            course_names.append(comp[0])
            lectures = sort_slide_names(os.listdir(os.path.join(slides_path, comp[0])))
            lnos.append(lectures.index('----'.join(comp[1:-1])))

    else:
        related_slides = []
    return len(disp_strs),filtered_related_slides,disp_strs,course_names,lnos,disp_colors,disp_snippets

def get_search_results(search):
    query = metapy.index.Document()
    query.content(search)
    print (query,idx,ranker,search)
    top_docs = ranker.score(idx, query, num_results=50)
    print (top_docs)
    top_docs = [slide_titles[x[0]] for x in top_docs]
    results = []
    disp_strs = []
    course_names = []
    snippets = []
    lnos = []
    top_slide_trim_names = []
    for r in top_docs:
        try:
            comp = r.split('##')
            disp_strs.append(' '.join(comp[0].replace('_','-').split('-')).title() + ' : ' + ' '.join(comp[-2].replace('.txt','').replace('_','-').split('-')).title() + ' , ' + ' '.join(comp[-1].replace('.pdf','').split('-')).title())
            course_names.append(comp[0])
            lectures = sort_slide_names(os.listdir(os.path.join(slides_path, comp[0])))
            lnos.append(lectures.index('----'.join(comp[1:-1])))
            if len(results) < 10:
                results.append(r)
            snippets.append(get_snippet_sentences(r, search))
        except ValueError:
            continue
    for x in range(len(results)):
        results[x] = results[x].replace('##', '----') + '.pdf'
    return len(results),results,disp_strs,course_names,lnos, snippets


if __name__ == '__main__':
    load_related_slides()
    print (related_dict)

