import os
import numpy as np
from sklearn.model_selection import train_test_split
from glob import glob

import SimpleITK as sitk
from transforms import *
import tensorflow as tf
import csv
import sys

import getpass
import warnings



def get_split(dataroot, data='sex'):
    # data = 'sex' sets sex as the label
    # anything else reverts to 'age' as label

    filedata            = np.load(dataroot+'bbdata.npy')
    subjects            = filedata[:,0]
    labels              = filedata[:,1] if data=='sex' else filedata[:,2]
    subjectfiles        = ['/data2/RCA2017/BB2964_full/{}/sa_ED.nii.gz'.format(s) for s in subjects]

    hist = np.histogram(labels, 2)

    newlabels           = []
    newsubjects         = []
    for i, label in enumerate(np.unique(labels)):
        newlabels.append(labels[labels==label])
        newsubjects.append(subjects[labels==label])
    newlabels = np.asarray(newlabels)
    newsubjects = np.asarray(newsubjects)


    if data=='sex':
        for i, part in enumerate(newlabels):
            newlabels[i] = np.eye(2)[newlabels[i]] # one-hot encoding for the sex


    seed = 42
    np.random.seed(seed)
    np.random.shuffle(newlabels)
    np.random.seed(seed)
    np.random.shuffle(newsubjects)

    mincounts = np.min(hist[0])
    totake = int(0.1*mincounts) #20% for testing 20% for validation and 60% for training

    X_test = np.concatenate([newsubjects[i][:totake] for i in range(0,len(np.unique(labels)))])
    X_val  =  np.concatenate([newsubjects[i][totake:2*totake] for i in range(0,len(np.unique(labels)))])
    X_train =  np.concatenate([newsubjects[i][2*totake:mincounts] for i in range(0,len(np.unique(labels)))])

    y_test = np.concatenate([newlabels[i][:totake] for i in  range(0,len(np.unique(labels)))])
    y_val  =  np.concatenate([newlabels[i][totake:2*totake] for i in  range(0,len(np.unique(labels)))])
    y_train =  np.concatenate([newlabels[i][2*totake:mincounts] for i in  range(0,len(np.unique(labels)))])

    return  X_train, y_train, X_val, y_val, X_test, y_test


def load_and_resample(ilist, dims, isseg=False):
    i_ = [resampleit(sitk.GetArrayFromImage(sitk.ReadImage(i)).transpose(1,2,0), dims, isseg) for i in ilist]

    return np.array([i for i in i_])

def get_masks(seg):
    ''' function to get the different classes as different channels in one-hot encoding format
    input seg shape = [h, w, d]
    output seg shape = [h, w, d, c] where c is number of classes
    one-hot-vector per pixel = [1 x c]
    '''
    nclass = len(np.unique(seg))
    s1, s2, s3 = seg.shape
    a = seg.reshape([np.prod(seg.shape)])
    b = np.zeros((np.prod(seg.shape),4))
    a[np.where(a==4)]=3         #make sure it's class 3 not class 4
    b[np.arange(b.shape[0]),a]=1

    seg = b.reshape([s1, s2, s3, 4])

    return seg


def cropbatch(images, segs, resample_dims = [128,128,8], maskout=False):
    x_ = np.zeros([images.shape[0]] + resample_dims)
    s_ = np.zeros([images.shape[0]] + resample_dims) #IF YOU WANT THE SEGS TOO
    for idx, im in enumerate(np.squeeze(images)):
        #seg         =  '/data2/RCA2017/BB2964_full/{}/label_sa_ED.nii.gz'.format(ids[idx])
        #seg         = resampleit(sitk.GetArrayFromImage(sitk.ReadImage(seg)).transpose(1,2,0), im.shape, True)
        seg = np.squeeze(segs[idx])
        xtmp, stmp  = cropit(im, seg=seg, margin=10)
        x_[idx,...]     = resampleit(xtmp, resample_dims, False)
        s_[idx,...]    = resampleit(stmp, resample_dims, True) #IF YOU WANT THE SEGS TOO

    if maskout:
    	x_[s_==0] = 0

    return np.expand_dims(x_,-1)



def dice(A, B):
    classes = np.unique(A)
    dsc = [2.0 * np.sum(np.logical_and(A==i, B==i)) / np.float((np.sum(A==i) + np.sum(B==i))) for i in classes]
    return dsc + [2.0 * np.sum(np.logical_and(A>0, B>0)) / np.float((np.sum(A>0) + np.sum(B>0)))]


def get_batch(ims, labels, numAugs, dims, dataroot='/data2/RCA2017', getsegs=True, cropem=False, maskout=False):
    
    imroot      = os.path.join(dataroot,'BB2964_full')
    segroot     = dataroot
    ifiles      = [os.path.join(imroot,str(subject),'sa_ED.nii.gz') for subject in ims]
    if getsegs:
        gfiles     = [os.path.join(imroot,str(subject),'label_sa_ED.nii.gz') for subject in ims]
        segs = load_and_resample(gfiles, dims, True)

    images  = load_and_resample(ifiles, dims)

    if numAugs !=0:
        augimages, augsegs   = do_aug(images, segs, numAugs, dims)

        auglabels = np.zeros([len(augimages),2])
        for idx in range(len(labels)):
            auglabels[idx*(numAugs+1):(idx+1)*(numAugs+1),:2] = labels[idx]
        return (cropbatch(augimages, augsegs, [128,128,8], maskout), auglabels) if cropem else (augimages, auglabels)
    else:
        images, segs = do_aug(images, segs, 0, dims, augem=0)
        return (cropbatch(images, segs, [128,128,8], maskout), labels) if cropem else (images, labels)


def do_aug(images, segs, numAugs, dims, augem=1):

    theseims = np.zeros([len(images)*numAugs+len(images)] +dims+[1])
    thesesegs = np.zeros([len(images)*numAugs+len(images)] +dims+[1])
    idim = 0
    for zz, im in enumerate(images):
        thisseg = segs[zz]
        thisim = im
        #thisim, thisseg = cropit(thisim, thisseg)
        #thisim = resampleit(im, dims) # comment this if not cropping
        #thisseg = resampleit(thisseg, dims, isseg=True)
        theseims[idim*(numAugs+1), :,:,:,0] = thisim
        thesesegs[idim*(numAugs+1), :,:,:,0] = thisseg
        #
        if augem == 1:
            idx=0           
        #    
            while idx < numAugs:
                idaug = (idim*(numAugs+1)) + idx + 1
                #
                theta   = 0.0
                factor  = 1.0
                offset  = 0.0
                axes = [0, 0]
                scalefactor = 1.0
                #
                thisim = im
                thisseg = segs[zz]              
                np.random.seed()
                numTrans = np.random.randint(2, 6, size=1)        
                allowedTrans = [0, 1, 2, 3, 4]
                whichTrans = np.random.choice(allowedTrans, numTrans, replace=False)
                #
                if 0 in whichTrans:
                    theta   = float(np.around(np.random.uniform(-15.0,15.0, size=1), 2))
                    thisim  = rotateit(thisim, theta)
                    thisseg = rotateit(thisseg, theta, isseg=True)
                #
                if 1 in whichTrans:
                    scalefactor  = float(np.around(np.random.uniform(0.85, 1.15, size=1), 2))
                    thisim  = scaleit(thisim, scalefactor)
                    thisseg = scaleit(thisseg, scalefactor, isseg=True)
                #
                #
                if 3 in whichTrans:
                    axes    = list(np.random.choice([0,1], 1, replace=True))
                    thisim  = flipit(thisim, axes+[0])
                    thisseg = flipit(thisseg, axes+[0])
                #
                if 4 in whichTrans:
                    offset  = list(np.random.randint(-8,8, size=2))
                    #currseg = thisseg
                    thisim  = translateit(thisim, offset)
                    thisseg = translateit(thisseg, offset, isseg=True)
                #
                if int(thisseg.sum())==0:
                    continue #if segmentation is blank, while loop carries on and adds another one without increasing idx                           
                #
                #thisim, thisseg = cropit(thisim, thisseg)
                thisim = resampleit(thisim, dims)
                thisseg = resampleit(thisseg, dims, isseg=True)
                #
                if int(thisseg.sum())== 0:
                    continue
                #
                if 2 in whichTrans:
                    factor  = float(np.around(np.random.uniform(0.8, 1.2, size=1), 2))
                    thisim  = intensifyit(thisim, factor)
                    #no intensity change on segmentation

                theseims[idaug, :,:,:,0] = thisim
                thesesegs[idaug, :,:,:,0] = thisseg
                #theseims[idaug, :,:,:,1:5] = get_masks(thisseg)
                #
                idx+=1
        #        
        idim+=1 
    return theseims, thesesegs
    
    
    
    
    
    
    
    
    
    