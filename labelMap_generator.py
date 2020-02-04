'''
Program used to create a CSV file showing mapping from index to class label.
This is needed to know the output of tflite.run() in Android.
We know that the output indices of tflite.run() correspond to the classes in alphabetical order.
Once we have the CSV file, we can move into the raw dir of the Android app
'''

import os, sys, csv
from glob import glob

def createClassNames(directory):
	'''
	input: directory list of all subdirectories
	output: hash map that maps indices to corresponding to classes
	'''

	class_names = [x.split("/")[-1] for x in glob(directory)]
	class_names = sorted(class_names)
	name_id_hashmap = dict(zip(class_names, range(len(class_names))))
	return name_id_hashmap

def main():
	labels = createClassNames('breed_images/*')
	with open('label_dictionary.csv', mode='w', newline="") as csv_file:
		writer = csv.writer(csv_file)
		for key, value in labels.items():
			writer.writerow([value, key])

if __name__ == '__main__':
	main()
