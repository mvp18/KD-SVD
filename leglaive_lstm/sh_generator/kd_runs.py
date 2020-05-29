import numpy as np

jobstr = "python3 main_kd.py -e 50 -es 7 -rd_lr 5 -bs 64 -lr 1e-4 "
jobs = []

for temp in [2, 5, 7, 10, 20]:
	for alpha in np.arange(0.1, 0.91, 0.1):						
		jobs.append(jobstr + 
		' '.join([
		'-temp ' + str(temp), \
		'-alpha ' + str(alpha), \
		'\n']))
												

f = open('../expts.sh', 'w')

for job in jobs:
	f.write(job)

f.close()

print("Number of jobs submitted: {}".format(len(jobs)))