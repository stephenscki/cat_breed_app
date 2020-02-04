'''
copy cat files and place them into their own directories of their breeds.
we do this so we can take advantage of keras' flow_from_dir option
'''


import os, shutil, re



def makeDirectoryOfBreed(breed):
	'''
	makes a directory of that breed inside the directory breed_images to store images, returns this new directory
	'''
	dest = 'cat_images_test/'+breed+'/'
	try:
		os.mkdir(dest)
	except: 
		print('directory already exists')
	return dest



def moveCatImages(directory):
	'''
	according to the README in annotations, cat image names are capitalized while dogs are not.
	so we can copy all the cats into their own directories in a new one
	'''
	with os.scandir(directory) as it:
		for entry in it:
			is_cat_name = re.sub('([a-zA-Z])', lambda x: x.groups()[0].upper(), entry.name, 1)
			if is_cat_name == entry.name:
				regexNum = re.search(r"\d", entry.name)
				breed = entry.name[:regexNum.start()-1]
				print("breed: " + breed)
				dest = makeDirectoryOfBreed(breed)
				dest_name = dest + entry.name
				shutil.copy(entry.path, dest_name)



def main():
	moveCatImages('images/')

if __name__ == '__main__':
	main()