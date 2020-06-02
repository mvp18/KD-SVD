import numpy as np

jobstr = "python3 main_kd.py -e 50 -es 5 -rd_lr 3 -bs 32 -lr 1e-4 -dr 0.2 "
jobs = []

for temp in [2, 5, 10, 20]:
	for alpha in np.arange(0.1, 0.91, 0.3):
		for fs in [32, 16, 4]:
			jobs.append(jobstr + 
			' '.join([
			'-fs ' + str(fs), \
			'-temp ' + str(temp), \
			'-alpha ' + str(alpha), \
			'\n']))
												

f = open('../expts.sh', 'w')

for job in jobs:
	f.write(job)

f.close()

print("Number of jobs submitted: {}".format(len(jobs)))