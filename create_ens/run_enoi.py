#!/usr/bin/python
#rewrite run_rttovmem.csh
import sys 
import os
import time
import numpy as np

##ddhh
ddhh = int(sys.argv[1])
exp = sys.argv[2]
codedir = sys.argv[3]

workdir = '/share/home/lililei1/feiyu/tc2019'
file1 = '{:s}/create_copy.csh'.format(codedir)
file2 = '{:s}/create_ensemble.csh'.format(codedir)

os.system('rm -r {:s}/enoi'.format(codedir))
os.system('mkdir -p {:s}/enoi'.format(codedir))

def write_bsub(mem):
	with open(file1,'r') as filecopy, open(file2,'w') as filegiirs:
		linels = 0
		for line in filecopy:
			linels +=1
			if linels==9:
				line = '#BSUB -o enoi/{:d}.out\n'.format(mem)
			elif linels==10:
				line = '#BSUB -e enoi/{:d}.out\n'.format(mem)
			elif linels==27:
				line = '@ ie = {:d}\n'.format(mem)

			filegiirs.write(line)

	#print 'Mem {:d} start'.format(mem)
	os.system('bsub -J "FeiyuENOI{:d}" < {:s}'.format(mem,file2))


if __name__ == '__main__':

	memgroup = np.arange(1,9)

	for mem in memgroup :
		if ( os.path.isfile('{:s}/{:s}/{:d}/advance_ensemble/{:d}/DONE'. \
			format(workdir, exp, ddhh, mem) ) == 0 ) :

			write_bsub(mem)
			time.sleep ( 1 )

	while ( np.min ( memgroup ) <= 72 ) :
		time.sleep ( 1 )
		for i in range(8) :
			if ( os.path.isfile ( '{:s}/{:s}/{:d}/advance_ensemble/{:d}/DONE'.\
				format(workdir, exp, ddhh, memgroup[i]) ) == 1 and memgroup[i] <= 72 ) :

				memgroup[i] += 8

				if ( os.path.isfile ( '{:s}/{:s}/{:d}/advance_ensemble/{:d}/DONE'.\
					format(workdir, exp, ddhh, memgroup[i]) ) == 0 ) :
					
					write_bsub( memgroup[i] )
					time.sleep ( 1 )


		if ( os.path.isfile ( '{:s}/{:s}/{:d}/advance_ensemble/79/DONE'.\
				format(workdir, exp, ddhh) ) == 1  and
			 os.path.isfile ( '{:s}/{:s}/{:d}/advance_ensemble/80/DONE'.\
				format(workdir, exp, ddhh) ) == 1 ) :

			 break


	print "ENOI Again"
	memgroup = np.arange(1,9)

	for mem in memgroup :
		if ( os.path.isfile('{:s}/{:s}/{:d}/advance_ensemble/{:d}/DONE'. \
			format(workdir, exp, ddhh, mem) ) == 0 ) :

			write_bsub(mem)
			time.sleep ( 1 )

	while ( np.min ( memgroup ) <= 72 ) :
		time.sleep ( 1 )
		for i in range(8) :
			if ( os.path.isfile ( '{:s}/{:s}/{:d}/advance_ensemble/{:d}/DONE'.\
				format(workdir, exp, ddhh, memgroup[i]) ) == 1 and memgroup[i] <= 72 ) :

				memgroup[i] += 8

				if ( os.path.isfile ( '{:s}/{:s}/{:d}/advance_ensemble/{:d}/DONE'.\
					format(workdir, exp, ddhh, memgroup[i]) ) == 0 ) :
					
					write_bsub( memgroup[i] )
					time.sleep ( 1 )


