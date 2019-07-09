# HEIMDAL

Heimdal.py is python script that watches and alerts user when mumax jobs are complete with a push notification. For this it uses the package 
notify-run.py and you must register your own notification channel through this module before heimdal.py will work. It is easy though just read 
the docs at `notify.run`

Heimdal.py pushes notifications when a simulation is done, giving the name and duration of the simulation. Internally, jobs are a class and they 
have two attributes, complete (boolean) and duration (numpy array). They are stored in a dictionary indexed by their file names. Every five 
seconds the dictionary and it's contents are updated. Jobs are detected by their output file and deemed complete when the duration file appears. 
When output files are moved or deleted from the local directory, their corresponding job objects are deleted. I hope this means that there are 
no memory leaks and therefore this python programme can run indefinitely in a tmux instance.

I hope this is useful for other mumax3 users to keep track of their workflow! (or anyone else doing computations that take too long to sit and 
watch...)
