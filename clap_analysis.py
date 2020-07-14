#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Modules required:
from pydub import AudioSegment as AS
import matplotlib.pyplot as plt
from pydub.playback import play
from scipy.io.wavfile import read
from scipy.signal import find_peaks
import numpy as np
import wave
import pylab
import statistics as st
import math
from scipy.io import wavfile
from scipy.fftpack import fft
AS.converter = r"*path to the file-*\ffmpeg.exe"
AS.ffmpeg = r"*path to the file-*\ffmpeg.exe"
AS.ffprobe =r"*path to the file-*\ffprobe.exe"


# In[ ]:


#This sectionof code is for converting files automatically from any audio format to wav
#Make sure to have only audio based files in the folder you give the function
#NOTE: All the audio files are required to be .wav file in all the functions thereof, and this function is made only
#to make it easier without any external source to do this.
import os
import python_utils as utils
import glob

folder = r"*path to the folder you have the files that you need to convert*"

def audio2wav(folder, verbose=True):
    #folder = utils.abs_path_dir(folder)
    filelist = glob.glob(folder)
    for index, entire_fn in enumerate(filelist):
        if verbose:
            print(str(index + 1) + "/" + str(len(filelist)) + " " + entire_fn)
        filen = entire_fn.split(".")[0]
        extension = entire_fn.split(".")[1]
        print(filen)
        print(extension)
        print(folder + entire_fn)
        print(folder + filen)
        audio = AS.from_file(entire_fn, format=extension)
        audio.export(filen + ".wav", format="wav")
    if verbose:
        print("Conversion done")


# In[ ]:


#Function used to find the position of claps in the function:
#input:
#clap_array - the variable stores the audio of the clap[narray]
#sampliFreq - Sampling frequency of the audio.[int]
#timeArray - the time over which the audio passed is[narray]
#output:
#A dictionary variable with:
# "clap_array" - audio of clap[narray]
# "clap_index" - indices in the clap_array where clap has occured[narray]
# "time_array" - the time over which the audio passed is[narray]
# "num_of_claps" - number of claps detected in the audio segment[int]

#Algorithm:
# step1: take cumulative sum of the whole absolute values of clap_array.[window has been refined through trial and error]
#        possibility to be refined further
# step2: find a threshold(mean of the cumulative sum of above array is used) and find all peaks in the cumulative sum
#        from above
# step3: the peaks found are further removed by finding a minimum height requirement for them(taken as 0.2 times the max
#        value in moving average.
# step4: define a minimum gap between claps, so that we can remove many of the extra peaks that happens close together
#        (min_gap is defined interms of maximum number of claps allowed in a second, you can change this to suit your exp)
# step5: The result is passed back (after counting the claps)

def clap_pos(clap_array,samplingFreq,timeArray):
    max_clap_sec = 5
    N = math.ceil(samplingFreq/(8000*max_clap_sec))
    cumsum, moving_aves = [0], []
    
    result = {
        "clap_array"   : clap_array,
        "clap_index"   : [],
        "time_array"   : timeArray,
        "num_of_claps" : []
    }
    #Step1:
    for i, x in enumerate(abs(clap_array), 1):
        cumsum.append(cumsum[i-1] + x)
        if i>=N:
            moving_ave = (cumsum[i] - cumsum[i-N])/N
            #can do stuff with moving_ave here
            moving_aves.append(moving_ave)
    
    #Step2:
    moving_aves = np.array(moving_aves)
    ####################################
    thresh = moving_aves.mean()
    indices = find_peaks(moving_aves, threshold=thresh)[0]
    ####################################
    
    #Step3:
    pos_indices = []
    min_height_indices = []
    corrected_indices = []
    ####redundant code##################
    for i in range(indices.__len__()):
        if moving_aves[indices[i]]>0:
            pos_indices.append(indices[i])
    ####################################
    
    if pos_indices.__len__()>0:
        min_height = 0.2*max(moving_aves[pos_indices])
    else:      
        return result

    for i in range(pos_indices.__len__()):
        if moving_aves[pos_indices[i]]>min_height:
            min_height_indices.append(pos_indices[i])
    #Step4:       
    #######finds the time_indice_ratio for min_gap######
    time_len = timeArray.__len__()
    if time_len > 0:
        time_indice_ratio = (timeArray[time_len-1]-timeArray[0])/time_len
    else:
        return result
    ####################################################
    

    min_gap = math.ceil(1/(max_clap_sec*time_indice_ratio))

    j = 0
    for i in range(min_height_indices.__len__()):
        temp = min_height_indices[i]
        if i==0:
            corrected_indices.append(temp)
        elif temp-corrected_indices[j] > min_gap:
            corrected_indices.append(temp)
            j += 1
    
    #Step5:
    
    num_of_claps = corrected_indices.__len__()
    
    
    result["clap_array"] = clap_array
    result["clap_index"] = corrected_indices
    result["time_array"] = timeArray
    result["num_of_claps"] = num_of_claps
            
    return result


# In[ ]:


#This is a function purely made for abstraction and all it really does is find difference between adjacent values in an 
#array, here it is used to find the indices difference between 2 claps(and thus the time difference, when you consider
#sampling frequency)
def clap_inter(clap_index):
    clap_interval = []
    num_of_claps = clap_index.__len__()
    for i in range(num_of_claps-1):
        clap_interval.append(clap_index[i+1]-clap_index[i])    
    return clap_interval


# In[ ]:


#A bit about the function:
#This is one of the earlier frunction used to check two claps passed to it is sychronised or not.
#This function was made early in the project and thus has a loose defintion of 'Synchronize',take it with a pinch of salt
#It takes two [narray]s which contain the index positions of claps(remember, we are tking both the clap audios on the
#same time line, so we dont have to worry about relative indices),we also pass the minimum time difference we are going
#to allow between two claps to assume they are in "sync" with each other.
#We also needs to take care that this function only finds, one single stretch of synchronised claps, i.e. if suppose a 
#claps goes out of sync in the middle, it wont be able to find the seond stretch once it synchronise again. So it is
#suggested to chop the array into smaller arrays so as to get better accuracies on the clap.
#Algorithm:
# Step1: we divide the claps into primary and secondary clap, primary is the clap that has clapped last,the other is
#        secondary
# Step2: Then we check if the difference between the last clap of the primary and secondary is lesser than the passed
#        min difference allowed, if not we chop off the last element of the primar clap and repeat step 2,until the
#        condition is satified or the array size has become 0(then we quit).[we also recheck which is primary and
#        secondary claps and reset them accordingly]
# Step3: After this we resize both arrays to the same length(with that of the one with shorter length)
#NOTE: This algorithm has obvious flaws, but most of this can be mitigated by cutting the data into smaller slices

#input:
#clap1 - narray - indices of clap1
#clap2 - narray - indices of clap2
#min_clap_syn - int - minimum difference allowed between claps indices.
#output:
#A Dictionary with following elements:
# "proc_clap1" - narray - array of the synced claps from clap1
# "proc_clap2" - narray - array of the synced claps from clap2
# "proc_intervals1" - narray - contains the time difference between the proc_clap1
# "proc_intervals2" - narray - contains the time difference between the proc_clap2

def sync_check(clap1,clap2,min_clap_syn):
    
    proc_data = {
        "proc_clap1" : [],
        "proc_clap2" : [],
        "proc_intervals1" : [],
        "proc_intervals2" : []
    }
    
    #Step1:
    clap1_intervals = clap_inter(clap1)
    clap2_intervals = clap_inter(clap2)
    if clap1.__len__()>0 and clap2.__len__()>0:
        if clap1[-1]>clap2[-1]:
            parity = 1
            prim_clap = clap1
            prim_clap_intervals = clap1_intervals
            sec_clap = clap2
            sec_clap_intervals = clap2_intervals
        else:
            parity = 0
            prim_clap = clap2
            prim_clap_intervals = clap2_intervals
            sec_clap = clap1
            sec_clap_intervals = clap1_intervals
    else:
        return proc_data
    proc_clap1 = [] 
    proc_clap2 = []
    proc_intervals1 = []
    proc_intervals2 = []
    
    
    while 1:
        val1 = prim_clap[-1]
        val2 = sec_clap[-1]
    
        if abs(val1-val2) < min_clap_syn:
            proc_clap1 = prim_clap
            proc_interval1 = prim_clap_intervals
            proc_clap2 = sec_clap
            proc_interval2 = sec_clap_intervals
            break
        elif prim_clap.__len__() > 1:
            prim_clap = prim_clap[0:-1]
            prim_clap_intervals = prim_clap_intervals[0:-1]
            if prim_clap[-1] < sec_clap[-1]:
                parity = parity^1
                temp = prim_clap
                prim_clap = sec_clap
                sec_clap = temp
            
                temp = prim_clap_intervals
                prim_clap_intervals = sec_clap_intervals
                sec_clap_intervals = temp
            
        else:
            #print("no synchronization")
            return proc_data
        
    #this is done so that we can make sure the right clap array is in the right processed one
    #(becomes useful when we input slices separately into the function)
    if parity^1:
        temp = proc_clap1
        proc_clap1 = proc_clap2
        proc_clap2 = temp          
        temp = proc_intervals1
        proc_intervals1 = proc_intervals2
        proc_intervals2 = temp
            
    #Step3:
    #making both the array equal sized
    len1 = proc_clap1.__len__()
    len2 = proc_clap2.__len__()
    if len1 > len2:
        proc_clap1 = proc_clap1[(len1-len2):]
        proc_intervals1 = proc_intervals1[(len2-len1):]
    else:
        proc_clap2 = proc_clap2[(len2-len1):]
        proc_intervals2 = proc_intervals2[(len2-len1):]

    proc_intervals1 = np.array(proc_intervals1)
    proc_intervals2 = np.array(proc_intervals2)
    num_of_syn_claps = 0
    for i in range(proc_intervals2.__len__()):
        if abs(proc_clap1[-(i+1)] - proc_clap2[-(i+1)])<=min_clap_sync:
            num_of_syn_claps += 1
        else:
            break
            
    
    proc_data["proc_clap1"] = proc_clap1[(proc_intervals2.__len__()-num_of_syn_claps):]
    proc_data["proc_clap2"] = proc_clap2[(proc_intervals2.__len__()-num_of_syn_claps):]
    proc_data["proc_intervals1"] = proc_intervals1[proc_intervals2.__len__()-num_of_syn_claps:]
    proc_data["proc_intervals2"] = proc_intervals2[proc_intervals2.__len__()-num_of_syn_claps:]
    
    
    return proc_data


# In[ ]:


#This function is used with above function to slice voice_arrays.
#input:
# addr - string - location of file with clap recording
#output:
#A Dictionary with following elements:
# "voice_slice" - narray - array of array(each inner array is the sliced voice)
# "time_slice" - corresponding time_slice
# "samplingFreq" - Sampling frequency of the file

def voice_slicing(addr):
    clap = AS.from_file(addr)
    ####################################
    clap_array = clap.get_array_of_samples()
    clap_array = np.array(clap_array)
    samplingFreq, mySound = wavfile.read(addr)
    samplePoints = float(mySound.shape[0])
    timeArray = np.arange(0, samplePoints, 1)
    timeArray = timeArray / samplingFreq
    ####################################
    time_slice_len = 5 #in sec
    index_len = 5*samplingFreq
    voice_slice = []
    time_slice = []
    temp = clap_array[0:clap_array.__len__():2]
    temp_time = timeArray
    
    while 1:
        leng = temp.__len__()
        if leng > index_len:
            voice_slice.append(temp[0:index_len])
            temp = temp[index_len:]
            time_slice.append(temp_time[0:index_len])
            temp_time = temp_time[index_len:]
        else:
            voice_slice.append(temp)
            time_slice.append(temp_time)
            break

    slices = {
        "voice_slice" : voice_slice,
        "time_slice"  : time_slice,
        "samplingFreq": samplingFreq
    }
    
    return slices
    


# In[ ]:


#this programme groups all the claps from the clap series with higher number of claps with
#one with lower number of claps:
#input:
# clap1 - indices of claps in first recording
# clap2 - indices of claps in second recording
#output:
# grouped_claps - narray - array of array with i th inner array being, the claps that are closest to it nd no other from
#                          the other clap
# parity - used to determine which was used as the primary and secondary array
#Algorithm:
# Step1: first make one clap(the one with lower number of claps) as the primar clap and the other as secondary
# Step2: The claps from secondary clap that are closest to i th clap(i.e.they are farther than this from other claps)
#        are grouped into i th array(element of another array)

def clap_group(clap1,clap2):
    len1 = clap1.__len__()
    len2 = clap2.__len__()
    parity = 0
    #Step1
    if len1<=len2:
        prim_claps = clap1
        sec_claps = clap2
    else:
        prim_claps = clap2
        sec_claps = clap1
        parity = 1
    
    grouped_claps = []
    
    #Step2:
    #first edge condition
    if prim_claps[0] > sec_claps[0]:
        j = 0
        temp_arr = []
        while sec_claps[j]<=(prim_claps[0]+prim_claps[1])/2:
            temp_arr.append(sec_claps[j])
            j += 1
        grouped_claps.append(temp_arr)
            
    else:
        grouped_claps.append([])
      
    #middle
    for i in range(prim_claps.__len__() - 2):
        temp_arr = []
        mid1 = (prim_claps[i] + prim_claps[i+1])/2
        mid2 = (prim_claps[i+1] + prim_claps[i+2])/2 
        j = 0
        
        for j in range(sec_claps.__len__()):
            if mid1<=sec_claps[j]<=mid2:
                temp_arr.append(sec_claps[j])

        grouped_claps.append(temp_arr)
    
    #second edge condition
    temp_arr = []
    for j in range(sec_claps.__len__()):
        if sec_claps[j]>= (prim_claps[-1]+prim_claps[-2])/2:
            temp_arr.append(sec_claps[j])
    
    grouped_claps.append(temp_arr)
    
    return grouped_claps,parity
        
        


# In[ ]:


#This function makes a plot that can intuitively tell you some features of relation between two claps.
#This function uses the clap_group function as part of it to do it.
#How to analyse the figure has been given in the report.
#input:
# fil1 - String - location of first recording
# fil2 - String - location of second recording
def clap_dif_group(fil1,fil2):
    file1 = fil1
    file2 = fil2
    clap1 = AS.from_file(file1)
    clap2 = AS.from_file(file2)

    samplingFreq, _ = wavfile.read(file1)

    clap_array1 = clap1.get_array_of_samples()[samplingFreq*2*10:-samplingFreq*2*5]#remember to change this later
    temp1 = clap_array1[0:clap_array1.__len__():2]
    clap_array2 = clap2.get_array_of_samples()[samplingFreq*2*10:-samplingFreq*2*5]#remember to change this later
    temp2 = clap_array2[0:clap_array2.__len__():2]

    samplePoints = float(temp1.__len__())
    timeArray = np.arange(0, samplePoints, 1)
    timeArray = timeArray / samplingFreq

    clap_indices1 = clap_pos(np.array(temp1),samplingFreq,timeArray)["clap_index"]
    clap_indices2 = clap_pos(np.array(temp2),samplingFreq,timeArray)["clap_index"]
    
    k,p = clap_group(clap_indices1,clap_indices2)
    
    n = 0
    if p == 1:
        clap_arr = clap_indices2
        clap_record = temp2
    else:
        clap_arr = clap_indices1
        clap_record = temp1
    sum_diff = 0
    leng = 0

    grouped_diff = []
    for i in range(clap_arr.__len__()):
        diff = []
        for j in range(k[i].__len__()):
            leng += 1
            diff.append(-(clap_arr[i]-k[i][j])/samplingFreq)
        if(k[i].__len__()!=0):
            n +=1
        sum_diff += sum(diff)
        grouped_diff.append(diff)

    clap_record = np.array(clap_record)

    print("number of claps with corresponding claps:",n)
    print("number of claps:",clap_arr.__len__())
    print(grouped_diff.__len__())
    print("Average difference between the grouped claps: ",sum_diff/leng)
    print(clap_arr)
    unique_claps = set()
    clap = []
    dif = []
    non_vals = []
    for i in range(grouped_diff.__len__()):
        for j in range(grouped_diff[i].__len__()):
            clap.append(i)
            dif.append(grouped_diff[i][j])
            unique_claps.add(i)
        if grouped_diff[i].__len__() == 0:
            clap.append(i)
            dif.append(None)
            non_vals.append(i)
            unique_claps.add(i)
        
        
    plt.figure(figsize=(20,10))
    for i in range(non_vals.__len__()):
        m = non_vals[i]
        plt.plot([m,m,m,m,m],range(-2,3))
    
    
    plt.scatter(clap[6:],dif[6:])
    plt.plot(np.zeros(unique_claps.__len__()))

    


# In[ ]:


#The following are experiments explained in the Report Appendix
#Only code is given here, as these are just application of above function.
def clap_exp1(clap_wav1,clap_wav2):
    
    
    audio_slices1 = voice_slicing(clap_wav1)
    audio_slices2 = voice_slicing(clap_wav2)
    
    num_of_slices = audio_slices1["voice_slice"].__len__() #both audio will be of the same length
    samplingFreq = audio_slices1["samplingFreq"]          #both audio has same sampling frequency
    timeArray = audio_slices1["time_slice"]
    size_of_slice = audio_slices1["voice_slice"][0].__len__()
    
    
    clap_index1 = []
    clap_index2 = []
    sync_slices1 = []
    sync_slices2 = []
    sync_slices_interval1 = []
    sync_slices_interval2 = []
    for i in range(num_of_slices):
        
        clap_array1 = audio_slices1["voice_slice"][i]
        clap_array2 = audio_slices2["voice_slice"][i]
        time_slice = timeArray[i]
        clap_info1 = clap_pos(clap_array1,samplingFreq,time_slice)
        clap_info2 = clap_pos(clap_array2,samplingFreq,time_slice)
        
        clap_index1.append(np.array(clap_info1["clap_index"])+i*(size_of_slice))
        clap_index2.append(np.array(clap_info2["clap_index"])+i*(size_of_slice))
        print("clap_pos1 ",i,":",clap_index1[-1])
        print("clap_pos2 ",i,":",clap_index2[-1])
        min_clap_syn = samplingFreq*0.5 #0.5 seconds
        
        syn_data = sync_check(clap_index1[i],clap_index2[i],min_clap_syn)
        sync_slices1.append(syn_data["proc_clap1"])
        sync_slices2.append(syn_data["proc_clap2"])
        sync_slices_interval1.append(syn_data["proc_intervals1"])
        sync_slices_interval2.append(syn_data["proc_intervals2"])
    print("samplingFreq:",samplingFreq)
        
 
        
    result = {
        "sync_slices1" : sync_slices1,
        "sync_slices2" : sync_slices2,
        "sync_slices_interval1" : sync_slices_interval1,
        "sync_slices_interval2" :sync_slices_interval2
    }
    return result

def exp2(median,clap_data):
    for i in range(clap_data.__len__()):
        clap_dif_group(median[i],clap_data[i])


def exp3(files):
    once = True
    samplingFreq = 0
    claps = []
    for file in files:
        temp_clap = AS.from_file(file)
        if once:
            samplingFreq,_ = wavfile.read(file)
            once = False
        clap_array = temp_clap.get_array_of_samples()
        temp = clap_array[0:clap_array.__len__():2]
        samplePoints = float(temp.__len__())
        timeArray = np.arange(0, samplePoints, 1)
        timeArray = timeArray / samplingFreq
        clap_indices = clap_pos(np.array(temp),samplingFreq,timeArray)["clap_index"]
        claps.append(clap_indices)
               
    
    num_of_channels = claps.__len__()
    
    fig = plt.figure()
    
    major_ticks_x = np.arange(0,90,9)
    major_ticks_y = np.arange(0,0.6,6)
    minor_ticks_x = np.arange(0,90,0.1)
    
    ax = fig.add_subplot(1,1,1)
    for i in range(num_of_channels):
        num_of_claps = claps[i].__len__()
        
        ax.set_xticks(major_ticks_x)
        ax.set_xticks(minor_ticks_x, minor = True)
        ax.set_yticks(major_ticks_y)
        
        y_temp = [i*0.1]*num_of_claps
        x_temp = [x/samplingFreq for x in claps[i]]
        ax.scatter(x_temp,y_temp)
    ax.grid(which = 'both')
    ax.show()

