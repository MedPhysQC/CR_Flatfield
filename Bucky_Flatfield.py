#!/usr/bin/env python
# PyWAD is an open-source set of plugins for the WAD-Software medical physics quality control software. 
# The WAD Software can be found on https://github.com/wadqc
# 
# The pywad package includes plugins for the automated analysis of QC images for various imaging modalities. 
# PyWAD has been originaly initiated by Dennis Dickerscheid (AZN), Arnold Schilham (UMCU), Rob van Rooij (UMCU) and Tim de Wit (AMC) 
#
#
# Date: 2018-11-15
# Version: 1.0
# Authors: C. den Harder
# Update to wad2: D. Dickerscheid
# Changelog:
#
#
# Description of this plugin:
# This plugin analyses an image of a homogeneous phantom in front of the tube,
# e.g.: 2mm Cu plate, or 20mm Al slab
# It produces DAP, REX, EI, SNR in the center ROI, and Homogeneity in 5 ROI's
#

__version__ = '20190427'
__author__ = 'c.den.harder'


import matplotlib
matplotlib.use('Agg') # Force matplotlib to not use any Xwindows backend.
import matplotlib.patches as patches

import os,sys
import numpy as np
import scipy 
import logging


import numpy as np
import tempfile
os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import pydicom as dicom
except ImportError:
    import dicom
from PIL import Image
from scipy import ndimage


from wad_qc.module import pyWADinput
from wad_qc.modulelibs import wadwrapper_lib



def acqdatetime_series(data, results, action):
    """
    Read acqdatetime from dicomheaders and write to IQC database

    Workflow:
        1. Read only headers
    """
    try:
        params = action['params']
    except KeyError:
        params = {}

    ## 1. read only headers
    dcmInfile = dicom.read_file(data.series_filelist[0][0], stop_before_pixels=True)

    dt = wadwrapper_lib.acqdatetime_series(dcmInfile)

    results.addDateTime('AcquisitionDateTime', dt)





def logTag():
    return "[Bucky_Flatfield] "


def filter_min_diff(inlist):
    x_min_diff = 10
    outlist = []
    if len(inlist) > 0:
        outlist.append(inlist[0])    
    return outlist

def diff_list(rowlist):
    return np.diff(rowlist)

# Clean DICOM string
def cleanstring(instring):
    forbidden = '[,]\'" '
    outstring = ''
    for ch in instring:
        if ch in forbidden:
            continue
        else:
            outstring += ch
    outstring = outstring.replace('UNUSED', '') # cleaning
    return outstring;

def analyse_image_quality(dcmInfile,pixeldataIn,prefix,params,results,level=None):
    '''
    Process a homogeneous image, and derive:
    - SNR
    - Homogeneity
    - Image thumbnail
    '''

    ######################################
    # Image quality parameters:          #
    # - SNR in central ROI               #
    # - homogeneity over 5 ROI's         #
    ######################################
    xsize,ysize = np.shape(pixeldataIn)

    ROIsize_pct = float(params.get('ROIsize [%]', 18))
    halfROIsize_x = round(ROIsize_pct*xsize/100/2)
    halfROIsize_y = round(ROIsize_pct*ysize/100/2)

    ctr_x_start = int(round((xsize/2)-halfROIsize_x));
    ctr_x_end = int(round((xsize/2)+halfROIsize_x));
    ctr_y_start = int(round((ysize/2)-halfROIsize_y));
    ctr_y_end = int(round((ysize/2)+halfROIsize_y));

    tl_x_start = int(round((xsize/4)-halfROIsize_x));
    tl_x_end = int(round((xsize/4)+halfROIsize_x));
    tl_y_start = int(round((ysize/4)-halfROIsize_y));
    tl_y_end = int(round((ysize/4)+halfROIsize_y));

    tr_x_start = int(round(((xsize/4)*3)-halfROIsize_x));
    tr_x_end = int(round(((xsize/4)*3)+halfROIsize_x));
    tr_y_start = int(round((ysize/4)-halfROIsize_y));
    tr_y_end = int(round((ysize/4)+halfROIsize_y));

    bl_x_start = int(round((xsize/4)-halfROIsize_x));
    bl_x_end = int(round((xsize/4)+halfROIsize_x));
    bl_y_start = int(round(((ysize/4)*3)-halfROIsize_y));
    bl_y_end = int(round(((ysize/4)*3)+halfROIsize_y));

    br_x_start = int(round(((xsize/4)*3)-halfROIsize_x));
    br_x_end = int(round(((xsize/4)*3)+halfROIsize_x));
    br_y_start = int(round(((ysize/4)*3)-halfROIsize_y));
    br_y_end = int(round(((ysize/4)*3)+halfROIsize_y));
    
    ctr_ROI = pixeldataIn[ctr_x_start:ctr_x_end,ctr_y_start:ctr_y_end];
    tl_ROI = pixeldataIn[tl_x_start:tl_x_end,tl_y_start:tl_y_end];
    tr_ROI = pixeldataIn[tr_x_start:tr_x_end,tr_y_start:tr_y_end];
    bl_ROI = pixeldataIn[bl_x_start:bl_x_end,bl_y_start:bl_y_end];
    br_ROI = pixeldataIn[br_x_start:br_x_end,br_y_start:br_y_end];
    
    ctr_ROI_mean = np.mean(ctr_ROI);
    tl_ROI_mean = np.mean(tl_ROI);
    tr_ROI_mean = np.mean(tr_ROI);
    bl_ROI_mean = np.mean(bl_ROI);
    br_ROI_mean = np.mean(br_ROI);

    ctr_ROI_std = np.std(ctr_ROI);
    tl_ROI_std = np.std(tl_ROI);
    tr_ROI_std = np.std(tr_ROI);
    bl_ROI_std = np.std(bl_ROI);
    br_ROI_std = np.std(br_ROI);
        
    ctr_ROI_SNR = ctr_ROI_mean / ctr_ROI_std;
    tl_ROI_SNR  = tl_ROI_mean  / tl_ROI_std;
    tr_ROI_SNR  = tr_ROI_mean  / tr_ROI_std;
    bl_ROI_SNR  = bl_ROI_mean  / bl_ROI_std;
    br_ROI_SNR  = br_ROI_mean  / br_ROI_std;

    results.addFloat(prefix + ' SNR', ctr_ROI_SNR)

    mean_ROI_mean = np.mean([ctr_ROI_mean,tl_ROI_mean,tr_ROI_mean,bl_ROI_mean,br_ROI_mean]);
    
    # WAD prescribes that each ROI should deviate 10% or less from the global mean.
    ctr_dev = (1-ctr_ROI_mean/mean_ROI_mean);
    tl_dev = (1-tl_ROI_mean/mean_ROI_mean);
    tr_dev = (1-tr_ROI_mean/mean_ROI_mean);
    bl_dev = (1-bl_ROI_mean/mean_ROI_mean);
    br_dev = (1-br_ROI_mean/mean_ROI_mean);

    max_dev = max([abs(ctr_dev),abs(tl_dev),abs(tr_dev),abs(bl_dev),abs(br_dev)]);
    results.addFloat(prefix + ' Uniformity', max_dev)

    #########################################
    # Image thunbnail for quick check       #
    # for presence of artifacts             #
    #########################################
    fig = plt.figure() 
    pt = pixeldataIn.transpose();


    plt.imshow(pt,cmap=plt.gray())

    rect = patches.Rectangle((ctr_x_start,ctr_y_start),halfROIsize_x*2,halfROIsize_y*2,linewidth=1,edgecolor='r',facecolor='none')
    plt.gca().add_patch(rect)

    rect = patches.Rectangle((tl_x_start,tl_y_start),halfROIsize_x*2,halfROIsize_y*2,linewidth=1,edgecolor='r',facecolor='none')
    plt.gca().add_patch(rect)

    rect = patches.Rectangle((tr_x_start,tr_y_start),halfROIsize_x*2,halfROIsize_y*2,linewidth=1,edgecolor='r',facecolor='none')
    plt.gca().add_patch(rect)

    rect = patches.Rectangle((bl_x_start,bl_y_start),halfROIsize_x*2,halfROIsize_y*2,linewidth=1,edgecolor='r',facecolor='none')
    plt.gca().add_patch(rect)

    rect = patches.Rectangle((br_x_start,br_y_start),halfROIsize_x*2,halfROIsize_y*2,linewidth=1,edgecolor='r',facecolor='none')
    plt.gca().add_patch(rect)

    label = wadwrapper_lib.readDICOMtag('0x0018,0x700A',dcmInfile)
    imageID = cleanstring(label)
    filename = prefix + '_unif_' + imageID + '.jpg'
    plt.savefig(filename)
    fig.clf()
    plt.close()
    results.addObject(os.path.splitext(filename)[0],filename)


def analyse_dose(dcmInfile,prefix,results):
    '''
    Process an image, and derive:
    - DAP
    - EI or REX
    - Additional DICOM tags
    '''
    slash = "/"

    
    ######################################
    # Dosis parameters from DICOM header #
    # - DAP                              #
    # - EI                               #
    # - REX                              #
    # - Exposure                         #
    ######################################
    DAP = -1;
    try:
        DAP = wadwrapper_lib.readDICOMtag('0x0018,0x115e',dcmInfile);
        #print DAP;
    except:
        print ('No Dose Area Product (DAP) available.')
    if ( DAP == '' ):
        print ('No Dose Area Product (DAP) available.')
        DAP = -1;
    results.addFloat(prefix + ' DAP', DAP)
    
    EI = -1;
    try:
        EI = wadwrapper_lib.readDICOMtag('0x0018,0x1411',dcmInfile);
        if ( EI == '' ):
            print ('No Exposure Index (EI) available.')
        else:    
            results.addFloat(prefix + ' EI', EI)
        #print EI;
    except:
        print ('No Exposure Index (EI) available.')

    REX = -1;
    try:    
        REX = wadwrapper_lib.readDICOMtag('0x0018,0x1405',dcmInfile);
        if ( REX == '' ):
            print ('No Reached Exposure value (REX) available.')
        else:
            results.addFloat(prefix + ' REX', REX)
        #print REX;
    except:
        print ('No Reached Exposure value (REX) available.')

    Exp_in_mAs = -1;
    try:    
        Exp_in_uAs = wadwrapper_lib.readDICOMtag('0x0018,0x1153',dcmInfile);
        Exp_in_mAs = 0.001*Exp_in_uAs;
    except:
        print ('No Exposure in uAs available.')
        Exp_in_mAs = -1;
    if ( Exp_in_mAs == '' ):
        print ('No Exposure in uAs available.')
        Exp_in_mAs = -1;

    if ( Exp_in_mAs < 0 ):
        #fetching the exposure in uAs failed        
        #try fetching Exposure in mAs instead
        try:
            Exp_in_mAs = wadwrapper_lib.readDICOMtag('0x0018,0x1152',dcmInfile);
        except:
            print ('No Exposure in mAs available.')
            Exp_in_mAs = -1;
        if ( Exp_in_mAs == '' ):
            print ('No Exposure in mAs available.')
            Exp_in_mAs = -1;
            
    results.addFloat(prefix + ' Exp.', Exp_in_mAs)

    #########################################
    # Protocol parameters from DICOM header #
    # - kVp                                 #
    # - SID                                 #
    # - Collimator                          #
    # - Grid                                #
    # - Filter                              #
    #########################################
    kVp = -1;
    try:    
        kVp = wadwrapper_lib.readDICOMtag('0x0018,0x0060',dcmInfile);
    except:
        print ('No tube voltage (kVp) available.')
    if ( kVp == '' ):
        print ('No tube voltage (kVp) available.')
        kVp = -1;
    results.addFloat(prefix + ' kVp', kVp)

    SID = -1;
    try:    
        SID = wadwrapper_lib.readDICOMtag('0x0018,0x1110',dcmInfile);
    except:
        print ('No Source to Detector (Image) distance (SID) available.')
    if ( SID == '' ):
        print ('No Source to Detector (Image) distance (SID) available.')
        SID = -1;
    results.addFloat(prefix + ' SID', SID)
 
    CollLV = -1;
    try:    
        CollLV = wadwrapper_lib.readDICOMtag('0x0018,0x1702',dcmInfile);
    except:
        print ('No Collimator Left Vertical position available.')
    if ( CollLV == '' ):
        print ('No Collimator Left Vertical position available.')
        CollLV = -1;
    results.addFloat(prefix + ' LV coll.', CollLV)

    CollRV = -1;
    try:    
        CollRV = wadwrapper_lib.readDICOMtag('0x0018,0x1704',dcmInfile);
    except:
        print ('No Collimator Right Vertical position available.')
    if ( CollRV == '' ):
        print ('No Collimator Right Vertical position available.')
        CollRV = -1;
    results.addFloat(prefix + ' RV coll.', CollRV)
 
    CollUH = -1;
    try:    
        CollUH = wadwrapper_lib.readDICOMtag('0x0018,0x1706',dcmInfile);
    except:
        print ('No Collimator Upper Horizontal position available.')
    if ( CollUH == '' ):
        print ('No Collimator Upper Horizontal position available.')
        CollUH = -1;
    results.addFloat(prefix + ' UH coll.', CollUH )
 
    CollLH = -1;
    try:    
        CollLH = wadwrapper_lib.readDICOMtag('0x0018,0x1708',dcmInfile);
    except:
        print ('No Collimator Lower Horizontal position available.')
    if ( CollLH == '' ):
        print ('No Collimator Lower Horizontal position available.')
        CollLH = -1;
    results.addFloat(prefix + ' LH coll.', CollLH)

    Grid = '';
    try:    
        Grid = wadwrapper_lib.readDICOMtag('0x0018,0x1166',dcmInfile);
    except:
        print ('No Grid info available.')
    results.addString(prefix + ' Grid', Grid)

    Filter = '';
    try:    
        Filter_array = wadwrapper_lib.readDICOMtag('0x0018,0x7050',dcmInfile);
        if ( type(Filter_array) is str ):
            Filter = Filter_array
        else:
            Filter = slash.join( Filter_array );
    except:
        print ('No Filter info available in 0x0018,0x7050 tag.')
        
    if ( Filter == '' ):
        #fetching the Filter failed        
        #try fetching Filter from 0x0018,0x1160 instead
        try:
            Filter_array = wadwrapper_lib.readDICOMtag('0x0018,0x1160',dcmInfile);
            if ( type(Filter_array) is str ):
                Filter = Filter_array
            else:
                Filter = slash.join( Filter_array );
        except:
            print ('No Filter info available.')
        if ( Filter == '' ):
            print ('No Filter info available.')
    results.addString(prefix + ' Filter', Filter)

    SWversion = '';
    try:    
        SWversion_array = wadwrapper_lib.readDICOMtag('0x0018,0x1020',dcmInfile);
        if ( type(SWversion_array) is str ):
            SWversion = SWversion_array
        else:
            SWversion = slash.join( SWversion_array );
    except:
        print ('No SWversion info available.')
    results.addString(prefix + ' SW', SWversion)

    DetectorID = '';
    try:    
        DetectorID = wadwrapper_lib.readDICOMtag('0x0018,0x700A',dcmInfile);
    except:
        print ('No Detector ID info available.')
    results.addString(prefix + ' Detector', DetectorID)

    DateLastCal = '';
    try:    
        DateLastCal = wadwrapper_lib.readDICOMtag('0x0018,0x700C',dcmInfile);
    except:
        print ('No Date of Last Calibration info available.')
    results.addString(prefix + ' Cal.', DateLastCal)


def getprotocolname(dcmInfile,params):
    protocoltag = '0x0018,0x1030' #default tag: ProtocolName
    if ("protocoltag" in sorted(params.keys())):
        protocoltag = params['protocoltag']

    protocolname = wadwrapper_lib.readDICOMtag(protocoltag,dcmInfile)

    return protocolname
    

def Bucky_Flatfield_main(data, results, action):
    print("Bucky image quality started");

    try:
        params = action['params']
    except KeyError:
        params = {}

    #inputfile = data.series_filelist[0]
    #Bucky_Flatfield_analyse_image(inputfile,results)
    for inputfile in data.series_filelist:
        print (inputfile)
        dcmInfile,pixeldataIn,dicomMode = wadwrapper_lib.prepareInput(inputfile,headers_only=True,logTag=logTag())
        

        ############################################################
        # Read protocol name, which will be used                   #
        # as prefix in the description column                      #
        # to distinguish the detectors and the ionisation chambers #
        ############################################################
        protocolname = getprotocolname(dcmInfile,params)
        print ("protocol name:", protocolname)
        prefix = cleanstring( protocolname )
        print ("prefix:", prefix)

        dcmInfile,pixeldataIn,dicomMode = wadwrapper_lib.prepareInput(inputfile,headers_only=False,logTag=logTag())
        analyse_image_quality(dcmInfile,pixeldataIn,prefix,params,results)

        ############################################################
	  # Dose assessment will be provided for all images          #
        ############################################################
        analyse_dose(dcmInfile,prefix,results)

        # One image per detector / ionisation chamber    
        # Separator between this and next image:

    return results


from wad_qc.module import pyWADinput
if __name__ == "__main__":
    data, results, config = pyWADinput()

    print(config)
    for name,action in config['actions'].items():

        if name == 'acqdatetime':
            acqdatetime_series(data, results, action)

        elif name == 'buckymain':
            Bucky_Flatfield_main(data, results, action)

    results.write()


