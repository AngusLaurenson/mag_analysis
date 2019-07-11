from notify_run import Notify
import os
import scipy as sp
import time

notify = Notify()

# notify.send("STARTED MONITORING MUMAX JOBS")
class job:
	def __init__(self):
		self.complete = False
		self.duration = 0

jobs={}

while True:
	job_list = [a for a in os.listdir('.') if a.endswith(".out") ]	
	for job_name in job_list:
		# check if all present jobs are logged in the jobs dict
		if job_name in jobs.keys():
			# if a job is already marked as done, do nothing
			if jobs[job_name].complete == True:
				pass
			# if a job is not done, check if it is done now
			# if it is done, send an alert, record duration and mark as done
			# if it is not done then do nothing
			else:
				fname = '/'.join((job_name,"duration"))
				try:
					jobs[job_name].duration = sp.loadtxt(fname)
					jobs[job_name].complete = True
					d = jobs[job_name].duration / 1e9
					day = d // (24 * 3600)
					d = d % (24 * 3600)
					hour = d // 3600
					d %= 3600
					minutes = d // 60
					d %= 60
					seconds = d            
					notify.send(f'MUMAX3: JOB COMPLETE \n {job_name}\nDuration = \
                    {day} days, \
                    {hour} hours, \
                    {minutes} minutes, \
                    {seconds} seconds')
				except:
					pass
		else:
			# if no entry exists for this job, 
			# create a job object in the jobs dict
			jobs[job_name] = job()
	time.sleep(5)
	# check if any old jobs are still in the jobs dictionary
	# if so, delete them to avoid a memory leak
	for key in list(jobs.keys()):
		if key not in job_list:
			del jobs[key]