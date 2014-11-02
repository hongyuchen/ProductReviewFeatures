import re

def get_classes(filename, acc_thresh, rev_thresh):

	classes = []

	doc_i = 0
	with open(filename, 'r') as f:
		line = f.readline()
		while line != '':
			if 'review/helpfulness: ' in line:
				numbers = re.findall("[0-9]+", line)
				num = float(numbers[0])
				denom = float(numbers[1])
				if denom > rev_thresh and (num)/(denom) > acc_thresh:
					classes.append(1)
				else:
					classes.append(-1)
				doc_i += 1
			line = f.readline()

	return classes

if __name__ == "__main__":
	classes = get_classes('small_train.txt', 0.7, 5)
	print classes
