import matplotlib.pyplot as plt
import codecs 
import numpy as np 
def main():
	with codecs.open('/Users/bhavya/Documents/CS510_proj/mooc-web-of-slides/src/slide_similarity/tmp/input.txt','r',encoding='utf-8') as f:
		txt = f.readlines()
	lens = []
	print (len(txt))
	for t in txt:
		lens.append(len(t.split()))
	plt.hist(lens)
	plt.show()
	lens = np.array(lens)
	print (lens[lens>256].shape)

if __name__ == '__main__':
	main()


