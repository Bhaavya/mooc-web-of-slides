from util import *

def build_corpi(courses_json_path,slide_names_file,concat_fields=None,concat_file=None,titles_file=None,text_file=None,subtitles_file=None):
	with open(courses_json_path,'r') as f:
		json_data = json.load(f)
	titles = []
	slides_cnt = []
	subtitles = []
	slide_names = []
	lens = []
	concat = []
	file_names = [concat_file,titles_file,text_file,subtitles_file]
	for course,lessons in json_data.items():
		for lesson,slides in lessons.items():
			for slide,slide_val in slides.items():
				if 'title' in slide_val.keys():
					titles.append(slide_val['title'])
				else:
					titles.append(' ')
				if 'lecture_transcript' in slide_val.keys():
					subtitles.append(slide_val['lecture_transcript'])
				else:
					subtitles.append(' ')
				if 'text' in slide_val.keys():
					slides_cnt.append(slide_val['text'])
				else:
					slides_cnt.append(' ')
				slide_names.append(lesson+'##'+slide)

	if concat_fields is not None:
		if 'titles' in concat_fields:
			concat = titles[:]
			titles = []
		if 'text' in concat_fields:
			if concat != []:
				concat = ["{} . {}".format(a_, b_).strip() for a_, b_ in zip(concat, slides_cnt)]
			else:
				concat = slides_cnt[:]
			slides_cnt = []
		if 'lecture_transcript' in concat_fields:
			if concat != []:
				concat = ["{} . {}".format(a_, b_).strip() for a_, b_ in zip(concat, subtitles)]
			else:
				concat = subtitles[:]
			subtitles = []
	for i,corpus in enumerate([concat,titles,slides_cnt,subtitles]):
		if corpus!=[]:
			write_utf_txt(corpus,file_names[i])

	write_utf_txt(slide_names,slide_names_file)

if __name__ == '__main__':
	build_corpi('/Users/bhavya/Documents/mooc-web-of-slides-local/src/slide_similarity/tmp/courses_json_preprocessed.json','/Users/bhavya/Documents/mooc-web-of-slides-local/src/slide_similarity/tmp/slides_names.txt',titles_file='/Users/bhavya/Documents/mooc-web-of-slides-local/src/slide_similarity/tmp/titles.txt',concat_fields=['text','lecture_transcript'],concat_file='/Users/bhavya/Documents/mooc-web-of-slides-local/src/slide_similarity/tmp/concat.txt',text_file='/Users/bhavya/Documents/mooc-web-of-slides-local/src/slide_similarity/tmp/slide_content.txt',subtitles_file='/Users/bhavya/Documents/mooc-web-of-slides-local/src/slide_similarity/tmp/subtitles.txt')