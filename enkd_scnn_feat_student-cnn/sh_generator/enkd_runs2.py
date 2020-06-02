import numpy as np

jobstr = "python3 main_kd.py -e 50 -es 5 -rd_lr 3 -bs 32 -lr 1e-4 -dr 0.2 "
jobs = []

for temp in [5, 10, 15, 20]:
	for alpha in np.arange(0.1, 0.91, 0.2):
		for fs in [16, 2]:
			for comb in ['am', 'gm']:
				jobs.append(jobstr + 
				' '.join([
				'-fs ' + str(fs), \
				'-temp ' + str(temp), \
				'-alpha ' + str(alpha), \
				'-comb ' + str(comb), \
				'\n']))
												

f = open('../expts2.sh', 'w')

for job in jobs:
	f.write(job)

f.close()

print("Number of jobs submitted: {}".format(len(jobs)))