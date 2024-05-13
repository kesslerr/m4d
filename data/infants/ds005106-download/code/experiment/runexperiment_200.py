
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: tgro5258
"""

from psychopy import core, event, visual, gui
import os,random,sys,math,json,requests,serial,time
from glob import glob
import pandas as pd
import numpy as np
from pylsl import StreamInfo, StreamOutlet

os.chdir(os.path.dirname(sys.argv[0]))

# debug things
debug_testsubject = 1
debug_usedummytriggers = 0
debug_windowedmode = 0
debug_save_screenshots = 0

#params
nrepeats = 20 #repeats per stimulus
nstimpersequence = 200 #stim per sequences

#timing (in secs)
refreshrate = 60
fixationduration = 1 - .5/refreshrate
stimduration = .1 - .5/refreshrate
isiduration = .2 - .5/refreshrate

#triggers
trigger_sequencestart = -2

stimuli = sorted(glob('stimuli/stim*.png'))
nstimuli = len(stimuli)
print('nstimuli:',nstimuli)
assert nstimuli==200, 'nstimuli should be 200'

if debug_testsubject:
    subjectnr = 0
else:
    # Get subject info
    subject_info = {'Subject number':''}
    if not gui.DlgFromDict(subject_info,title='Enter subject info:').OK:
        print('User hit cancel at subject information')
        exit()
    try:
        subjectnr = int(subject_info['Subject number'])
    except:
        raise

outfn = 'sub-%02i_task-fix_events.csv'%subjectnr
if not debug_testsubject and os.path.exists(outfn):
    raise Exception('%s exists'%outfn)
random.seed(subjectnr)

stimnumber=[]
for i in range(nrepeats):
    stimnumber += random.sample(range(nstimuli),nstimuli)
    
eventlist = pd.DataFrame(stimnumber,columns=['stimnumber'])
eventlist['sequencenumber'] = [math.floor(x/nstimpersequence) for x in range(len(eventlist))]
eventlist['blocksequencenumber'] = [math.floor(x/nstimuli) for x in range(len(eventlist))]
eventlist['presentationnumber'] = [x%nstimpersequence for x in range(len(eventlist))]
eventlist['stim'] = [stimuli[i] for i in eventlist['stimnumber']]
eventlist['stimnumber'] = [i+1 for i in eventlist['stimnumber']]
nsequences = int(nstimuli*nrepeats/nstimpersequence)

def writeout(eventlist):
    with open(outfn,'w') as out:
        eventlist.to_csv(out,index_label='eventnumber')

writeout(eventlist)

# =============================================================================
# %% START
# =============================================================================
if debug_windowedmode:
    win=visual.Window([700,700],units='pix')
else:
    win=visual.Window(units='pix',fullscr=True)
mouse = event.Mouse(visible=False)

fixation = visual.ImageStim(win, 'Fixation.png', size=16,
        name='fixation', autoLog=False)
fixationtarget = visual.ImageStim(win, 'Fixationtarget.png', size=16,
        name='fixationtarget', autoLog=False)

querytext = visual.TextStim(win,text='',pos=(0,200),name='querytext')
progresstext = visual.TextStim(win,text='',pos=(0,100),name='progresstext')
sequencestarttext = visual.TextStim(win,text='',pos=(0,50),name='sequencestarttext')
trigger = visual.Rect(win,size=.01,units='norm',pos=(-1,1),name='triggersquare',fillColorSpace='rgb255',lineWidth=0)

filesep='/'
if sys.platform == 'win32':
    filesep='\\'
    
def check_abort(k):
    if k and k[0][0]=='q':
        writeout(eventlist)
        raise Exception('User pressed q')
        
screenshotnr = 0
def take_screenshot(win):
    global screenshotnr 
    screenshotnr += 1
    win.getMovieFrame()
    win.saveMovieFrames('screenshots/screen_%05i.png'%screenshotnr)
                
def loadstimtex(stimname):
    return visual.ImageStim(win,stimname,size=375,name=stimname.split(filesep)[-1])

# first create a new stream info (here we set the name to BioSemi,
# the content-type to EEG, 8 channels, 100 Hz, and float-valued data) The
# last value would be the serial number of the device or some other more or
# less locally unique identifier for the stream as far as available (you
# could also omit it but interrupted connections wouldn't auto-recover).
triggerportinfo = StreamInfo('python', 'Markers', 1, 1000, 'int32', 'pythonEEGmarcs')
triggerport = StreamOutlet(triggerportinfo)

def sendtrigger(trigger_value):
    triggerport.push_sample([trigger_value])

# def sendtrigger(trigger_value):
#     trigger.fillColor = [trigger_value, 0, 0]
#     trigger.draw()

nevents = len(eventlist)
sequencenumber = -1
for eventnr in range(nevents):
    first = eventlist['sequencenumber'].iloc[eventnr]>sequencenumber
    if first: #start of sequence
        writeout(eventlist)
        
        sequencenumber = eventlist['sequencenumber'].iloc[eventnr]
                    
        fixation.draw()
        win.flip()
        nstimthissequence = sum(eventlist['sequencenumber']==sequencenumber)
        stimtex = []
        for i in range(nstimthissequence):
            s = loadstimtex(eventlist['stim'].iloc[eventnr+i])
            stimtex.append(s)
                        
        progresstext.text = '%i / %i'%(1+sequencenumber,nsequences)
        progresstext.draw()
        sequencestarttext.text = 'Press 3 to start the next block'
        sequencestarttext.draw()
        fixation.draw()
        time_fixon = win.flip()
        k=event.waitKeys(keyList='3q', modifiers=False, timeStamped=True)
        check_abort(k)
        vv = 1.05
        while True:
            k=event.getKeys(keyList='3q', modifiers=False, timeStamped=True)
            if k:
                check_abort(k)
                break
            fixation.size = fixation.size * vv
            if fixation.size[0]>500:
                vv=0.95
            if fixation.size[0]<10:
                vv=1.05
            fixation.draw()
            win.flip()
        fixation.size = 16
        fixation.draw()
        sendtrigger(trigger_sequencestart)
        time_fixon = win.flip()
        while core.getTime() < time_fixon + fixationduration:pass
    
    response=0
    rt=0
    stimnum = eventlist['presentationnumber'].iloc[eventnr]
    stim = stimtex[stimnum]
    stimname = stim.name
    stim.draw()
    fixation.draw()
    sendtrigger(eventlist['stimnumber'].iloc[eventnr])
    time_stimon=win.flip()
    if debug_save_screenshots:take_screenshot(win)
    
    while core.getTime() < time_stimon + stimduration:pass
    
    fixation.draw()
    sendtrigger(-1)
    time_stimoff=win.flip()
    if debug_save_screenshots:take_screenshot(win)
            
    eventlist.at[eventnr, 'stimname'] = stimname
    eventlist.at[eventnr, 'time_stimon'] = time_stimon
    eventlist.at[eventnr, 'time_stimoff'] = time_stimoff
    eventlist.at[eventnr, 'stimdur'] = time_stimoff-time_stimon

    #get response
    k=event.getKeys(keyList='123q', modifiers=False, timeStamped=True)
    if k:
        check_abort(k)
        fixationtarget.draw()
        sequencestarttext.text = 'PAUSED'
        sequencestarttext.draw()
        win.flip()
        while 1:
            k=event.getKeys(keyList='123q', modifiers=False, timeStamped=True)
            if k:
                check_abort(k)
                break
    while core.getTime() < time_stimon + isiduration:pass

writeout(eventlist)
print(str(sys.exc_info()))
sequencestarttext.text='Experiment finished!'
sequencestarttext.draw()
win.flip()
core.wait(1)
win.close()
exit()
