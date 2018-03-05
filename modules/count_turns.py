import os 

data_dir = 'D:\\IEMOCAP_full_release\\'

for i in range(1,6):
	sess_dir = data_dir + 'Session' + str(i) + '\\sentences\\\wav\\'
	
	file_count = 0 
	for subdir, dirs, files in os.walk(sess_dir):

		for file in files: 
			if file.endswith('wav'):
				file_count += 1

	print('{} files in session {}'.format(file_count, i))

