import os
import cv2 as cv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


### FOLDER SETUP

path_original =  'D:/Uni/9.9 Thesis/data/original/full/'
path_processed = 'D:/Uni/9.9 Thesis/data/processed/'
path_norm =      'D:/Uni/9.9 Thesis/data/processed/full_normalized/'

path_maskP =     'D:/Uni/9.9 Thesis/data/processed/mask_prostate/'
path_maskC =     'D:/Uni/9.9 Thesis/data/processed/mask_cancer/'

path_var   =      'D:/Uni/9.9 Thesis/data/processed/full_variance/'


### GET METADATA

metadata_path = []
metadata_patient = []
metadata_cycle = []
metadata_slice = []
metadata_height = []
metadata_width = []
metadata_intmin = []
metadata_intavg = []
metadata_intmax = []

for patient_i in os.listdir(path_original):
    
    for cycle_i in os.listdir(f'{path_original}{patient_i}'):
        if cycle_i.startswith("contrast-t"):
            
            print(f'>> READING METADATA > PATIENT {patient_i} CYCLE {cycle_i}')
            
            for slice_i in os.listdir(f'{path_original}{patient_i}/{cycle_i}'):
                image = cv.imread(f'{path_original}{patient_i}/{cycle_i}/{slice_i}', cv.IMREAD_ANYDEPTH)
                
                metadata_path.append(f'{patient_i}/{cycle_i}/{slice_i}')
                metadata_patient.append(int(patient_i.lstrip('0')))
                metadata_cycle.append(int(cycle_i.replace('contrast-t', '')))
                metadata_slice.append(int(os.path.splitext(slice_i)[0]))
                
                metadata_height.append(image.shape[0])
                metadata_width.append(image.shape[1])
                
                metadata_intmin.append(np.min(image))
                metadata_intavg.append(np.round(np.mean(image), decimals = 2))
                metadata_intmax.append(np.max(image))

metadata = pd.DataFrame({
    'path':     metadata_path,
    'patient':  metadata_patient,
    'cycle':    metadata_cycle,
    'slice':    metadata_slice,
    'height':   metadata_height,
    'width':    metadata_width,
    'intmin':   metadata_intmin,
    'intavg':   metadata_intavg,
    'intmax':   metadata_intmax
    })


metadata.to_excel(f'{path_processed}metadata.xlsx')
metadata.to_csv(f'{path_processed}metadata.csv')


### RESCALE PROSTATE MASKS

for patient_i in os.listdir(path_original):
    
    patient_i_rename = int(patient_i.lstrip('0')) # patient name not from metadata, needs manual name adjustment

    for cycle_i in os.listdir(f'{path_original}{patient_i}'):
        if cycle_i.startswith("prostateMask"):
            
            print(f'>> READING MASK > PATIENT {patient_i} CYCLE {cycle_i}')
            
            for slice_i in os.listdir(f'{path_original}{patient_i}/{cycle_i}'):
                image = cv.imread(f'{path_original}{patient_i}/{cycle_i}/{slice_i}', cv.IMREAD_ANYDEPTH)
                
                image_normalized = image * 255
                
                if not os.path.exists(f'{path_maskP}{patient_i_rename}'): # if folder doesnt exist - create
                    os.makedirs(f'{path_maskP}{patient_i_rename}')

                # print(f'>> SAVING FILE > PATIENT {patient_i_rename} SLICE {slice_i}')
            
                cv.imwrite(f'{path_maskP}{patient_i_rename}/{slice_i}', image_normalized)


### RESCALE CANCER MASKS

for patient_i in os.listdir(path_original):
    
    patient_i_rename = int(patient_i.lstrip('0')) # patient name not from metadata, needs manual name adjustment

    for cycle_i in os.listdir(f'{path_original}{patient_i}'):
        if cycle_i.startswith("regionMask"):
            
            print(f'>> READING MASK > PATIENT {patient_i} CYCLE {cycle_i}')
            
            for slice_i in os.listdir(f'{path_original}{patient_i}/{cycle_i}'):
                image = cv.imread(f'{path_original}{patient_i}/{cycle_i}/{slice_i}', cv.IMREAD_ANYDEPTH)
                
                image_normalized = image * 255
                
                if not os.path.exists(f'{path_maskC}{patient_i_rename}'): # if folder doesnt exist - create
                    os.makedirs(f'{path_maskC}{patient_i_rename}')

                # print(f'>> SAVING FILE > PATIENT {patient_i_rename} SLICE {slice_i}')
            
                cv.imwrite(f'{path_maskC}{patient_i_rename}/{slice_i}', image_normalized)


### NORMALIZE IMAGES

metadata = pd.read_csv(f'{path_processed}metadata.csv')

total_patients = len(np.unique(metadata['patient']))
print(f'Total patients: {total_patients}')

norm_intavg = np.zeros(shape = [len(metadata['path'])])
norm_intstd = np.zeros(shape = [len(metadata['path'])])

for patient_i in np.unique(metadata['patient']):
    
    bool_patient = np.equal(metadata['patient'], patient_i)
    intmax_patient = max(metadata['intmax'] * bool_patient) # select max intensity for given patient
    
    for enum_index, enum_value in enumerate(metadata['patient']):
        
        if enum_value == patient_i: # only for given patient
            
            image_cycle = metadata['cycle'][enum_index]
            image_slice = metadata['slice'][enum_index]
            
            image = cv.imread(f'{path_original}{metadata_path[enum_index]}', cv.IMREAD_ANYDEPTH)
            image_normalized = image / intmax_patient * 255
            
            norm_intavg[enum_index] = np.mean(image_normalized)
            norm_intstd[enum_index] = np.std(image_normalized)
            
            if not os.path.exists(f'{path_norm}{patient_i}'): # if folder doesnt exist - create
                os.makedirs(f'{path_norm}{patient_i}')
            
            print(f'>> SAVING FILE > PATIENT {patient_i} CYCLE {image_cycle} SLICE {image_slice}')
            
            # SAVE SPECIFIC SLICE:
            # if image_slice == 16:
            #     cv.imwrite(f'{path_norm}{patient_i}/{image_cycle}-{image_slice}.png', image_normalized)
            
            # SAVE ALL SLICES:
            cv.imwrite(f'{path_norm}{patient_i}/{image_cycle}-{image_slice}.png', image_normalized)


### VARIANCE MAP

metadata = pd.read_csv(f'{path_processed}metadata.csv')

for patient_i in os.listdir(path_norm):
    # print(f'Processing patient {patient_i}')
    
    bool_patient = np.equal(metadata['patient'], int(patient_i))
    height_patient = max(metadata['height'] * bool_patient) # select max height for given patient
    width_patient = max(metadata['width'] * bool_patient) # select max width for given patient
    
    filenames = os.listdir(f'{path_norm}{patient_i}')
    slices = [int(k.split('-')[1]) for k in [i.split('.')[0] for i in filenames]]
    
    for slice_i in np.unique(slices):
        print(f'PATIENT {patient_i} SLICE {slice_i}')
             
        slice_stack = np.zeros(shape = [height_patient, width_patient])
        slice_stack_blank = True

        for enum_index, enum_value in enumerate(slices):
            
            if enum_value == slice_i:
                
                if slice_stack_blank:
                    slice_stack = cv.imread(f'{path_norm}{patient_i}/{filenames[enum_index]}', cv.IMREAD_ANYDEPTH)
                    slice_stack_blank = False
                else:
                    new_slice = cv.imread(f'{path_norm}{patient_i}/{filenames[enum_index]}', cv.IMREAD_ANYDEPTH)
                    slice_stack = np.dstack((slice_stack, new_slice))
        
        if len(slice_stack.shape) == 2: # if number of stacks = 1 (only one cycle for slice), then zero standard deviation
            slice_stack_std = np.zeros(shape = slice_stack.shape)
        else:
            slice_stack_std = np.std(slice_stack, axis = 2)
        
        # slice_stack_std_norm = slice_stack_std / np.max(slice_stack_std) * 255 # normalize
        slice_stack_std_norm = slice_stack_std * 4 # normalize
        
        if not os.path.exists(f'{path_var}{patient_i}'): # if folder doesnt exist - create
            os.makedirs(f'{path_var}{patient_i}')
        
        cv.imwrite(f'{path_var}{patient_i}/{slice_i}.png', slice_stack_std)
        cv.imwrite(f'{path_var}{patient_i}/norm {slice_i}.png', slice_stack_std_norm)



### INTENSITY CURVES v3 mean/quantile/minmax

img_type        = []
patient_id      = []
cycle_id        = []
slice_id        = []
mask_int_mean   = []
mask_int_q10    = []
mask_int_q90    = []
mask_int_min    = []
mask_int_max    = []

for patient_i in os.listdir(f'{path_processed}full_normalized'):
    
    print(f'CALCULATING PATIENT {patient_i}')
    
    for cycle_slice_i in os.listdir(f'{path_processed}full_normalized/{patient_i}'):
        
        filename_cycle = os.path.splitext(cycle_slice_i)[0].split('-')[0]
        filename_slice = os.path.splitext(cycle_slice_i)[0].split('-')[1]
        
        if os.path.isfile(f'{path_processed}mask_prostate/{patient_i}/{filename_slice}.png') & \
           os.path.isfile(f'{path_processed}mask_cancer/{patient_i}/{filename_slice}.png'): # check if mask exists for this cycle/slice
            
            contrast_img = cv.imread(f'{path_processed}full_normalized/{patient_i}/{cycle_slice_i}')
            prostate_img = cv.imread(f'{path_processed}mask_prostate/{patient_i}/{filename_slice}.png')
            cancer_img = cv.imread(f'{path_processed}mask_cancer/{patient_i}/{filename_slice}.png')
            
            img_type.append("prostate")
            patient_id.append(int(patient_i))
            cycle_id.append(int(filename_cycle))
            slice_id.append(int(filename_slice))
            
            selected_region = contrast_img[np.where((prostate_img == 255) & (cancer_img != 255))]
            if len(selected_region) == 0:
                mask_int_mean.append(np.nan)
                mask_int_q10.append(np.nan)
                mask_int_q90.append(np.nan)
                mask_int_min.append(np.nan)
                mask_int_max.append(np.nan)
            else:
                mask_int_mean.append(np.mean(selected_region))
                mask_int_q10.append(np.quantile(selected_region, 0.10))
                mask_int_q90.append(np.quantile(selected_region, 0.90))
                mask_int_min.append(np.min(selected_region))
                mask_int_max.append(np.max(selected_region))
                
            img_type.append("cancer")
            patient_id.append(int(patient_i))
            cycle_id.append(int(filename_cycle))
            slice_id.append(int(filename_slice))
            
            selected_region = contrast_img[np.where(cancer_img == 255)]
            if len(selected_region) == 0:
                mask_int_mean.append(np.nan)
                mask_int_q10.append(np.nan)
                mask_int_q90.append(np.nan)
                mask_int_min.append(np.nan)
                mask_int_max.append(np.nan)
            else:
                mask_int_mean.append(np.mean(selected_region))
                mask_int_q10.append(np.quantile(selected_region, 0.10))
                mask_int_q90.append(np.quantile(selected_region, 0.90))
                mask_int_min.append(np.min(selected_region))
                mask_int_max.append(np.max(selected_region))

curves = pd.DataFrame({
    'img_type':     img_type,
    'patient_id':     patient_id,
    'cycle_id':   cycle_id,
    'slice_id':   slice_id,
    'mask_int_mean':   mask_int_mean,
    'mask_int_q10': mask_int_q10,
    'mask_int_q90': mask_int_q90,
    'mask_int_min': mask_int_min,
    'mask_int_max': mask_int_max
    })

curves.to_excel(f'{path_processed}curves_chart_prostate-vs-cancer-masks-v3.xlsx')
curves.to_csv(f'{path_processed}curves_chart_prostate-vs-cancer-masks-v3.csv')

curves = pd.read_csv(f'{path_processed}curves_chart_prostate-vs-cancer-masks-v3.csv')

if not os.path.exists(f'{path_processed}chart_prostate-vs-cancer-masks-v3/'): # if folder doesnt exist - create
    os.makedirs(f'{path_processed}chart_prostate-vs-cancer-masks-v3/')

for patient_id in curves['patient_id'].unique():
    print(f'>> GRAPHING PATIENT {patient_id}')
    
    fig = plt.figure(dpi=150, figsize=(14,8))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    ax = fig.add_subplot(2, 3, 1)
    sns.lineplot(data=curves.query(f'patient_id == {patient_id}'),
                 x="cycle_id",
                 y="mask_int_mean",
                 hue="img_type",
                 units="slice_id",
                 estimator=None, lw=1).set_title(f'patient_id={patient_id}')
    ax = fig.add_subplot(2, 3, 2)
    sns.lineplot(data=curves.query(f'patient_id == {patient_id}'),
                 x="cycle_id",
                 y="mask_int_q10",
                 hue="img_type",
                 units="slice_id",
                 estimator=None, lw=1).set_title(f'patient_id={patient_id}')
    ax = fig.add_subplot(2, 3, 3)
    sns.lineplot(data=curves.query(f'patient_id == {patient_id}'),
                 x="cycle_id",
                 y="mask_int_q90",
                 hue="img_type",
                 units="slice_id",
                 estimator=None, lw=1).set_title(f'patient_id={patient_id}')
    ax = fig.add_subplot(2, 3, 5)
    sns.lineplot(data=curves.query(f'patient_id == {patient_id}'),
                 x="cycle_id",
                 y="mask_int_min",
                 hue="img_type",
                 units="slice_id",
                 estimator=None, lw=1).set_title(f'patient_id={patient_id}')
    ax = fig.add_subplot(2, 3, 6)
    sns.lineplot(data=curves.query(f'patient_id == {patient_id}'),
                 x="cycle_id",
                 y="mask_int_max",
                 hue="img_type",
                 units="slice_id",
                 estimator=None, lw=1).set_title(f'patient_id={patient_id}')
    plt.savefig(f'{path_processed}chart_prostate-vs-cancer-masks-v3/{patient_id}.png', dpi=150)
    # plt.clf()
    plt.close()


### SLIC IMPLEMENTATION

plt.hist(np.ravel(img),bins=100)

np.quantile(img, 0.01)
np.quantile(img, 0.5)
np.quantile(img, 0.99)

from skimage.segmentation import slic, mark_boundaries

img = cv.imread(f'{path_var}50/12.png')
img = cv.imread(f'{path_var}50/norm 12.png')
cv.imshow('', img)

segments = slic(img, n_segments=100, compactness=5, start_label=1)
cv.imshow('', mark_boundaries(img, segments, color=(0,0,1)))

msk = cv.imread(f'{path_maskP}50/12.png', cv.IMREAD_GRAYSCALE)
cv.imshow('', msk)

segments = slic(img, n_segments=20, compactness=5, start_label=1, mask=msk)
cv.imshow('', mark_boundaries(img, segments, color=(0,0,1)))
cv.imwrite(f'{path_processed}1.png', mark_boundaries(img, segments, color=(0,0,1))*255)

# patient 50

img = cv.imread(f'{path_norm}50/25-12.png')
segments = slic(img, n_segments=200, compactness=5, start_label=1)
cv.imshow('', mark_boundaries(img, segments, color=(0,0,1)))

cv.imwrite(f'{path_processed}out1.png', img2)

# patient 50 all image
# same cycle, different slices
img = cv.imread(f'{path_norm}50/25-11.png')
segments = slic(img, n_segments=200, compactness=5, start_label=1)
cv.imwrite(f'{path_processed}25-11.png', mark_boundaries(img, segments, color=(0,0,1))*255)
img = cv.imread(f'{path_norm}50/25-12.png')
segments = slic(img, n_segments=200, compactness=5, start_label=1)
cv.imwrite(f'{path_processed}25-12.png', mark_boundaries(img, segments, color=(0,0,1))*255)
img = cv.imread(f'{path_norm}50/25-13.png')
segments = slic(img, n_segments=200, compactness=5, start_label=1)
cv.imwrite(f'{path_processed}25-13.png', mark_boundaries(img, segments, color=(0,0,1))*255)
# different cycles, same slice
img = cv.imread(f'{path_norm}50/24-12.png')
segments = slic(img, n_segments=200, compactness=5, start_label=1)
cv.imwrite(f'{path_processed}24-12.png', mark_boundaries(img, segments, color=(0,0,1))*255)
img = cv.imread(f'{path_norm}50/25-12.png')
segments = slic(img, n_segments=200, compactness=5, start_label=1)
cv.imwrite(f'{path_processed}25-12.png', mark_boundaries(img, segments, color=(0,0,1))*255)
img = cv.imread(f'{path_norm}50/26-12.png')
segments = slic(img, n_segments=200, compactness=5, start_label=1)
cv.imwrite(f'{path_processed}26-12.png', mark_boundaries(img, segments, color=(0,0,1))*255)

# patient 50 prostate only
img = cv.imread(f'{path_norm}50/25-11.png')
mask = cv.imread(f'{path_processed}mask_prostate/50/11.png', cv.IMREAD_GRAYSCALE) #cv.IMREAD_GRAYSCALE
segments = slic(img, n_segments=10, compactness=3, start_label=1, mask=mask)
cv.imshow('', mark_boundaries(img, segments, color=(0,0,1)))

cv.imshow('', img)
cv.imshow('', segments)
#img = cv.bitwise_and(img, mask)




# separating segments


img_type        = []
patient_id      = []
cycle_id        = []
slice_id        = []
mask_int_mean   = []
mask_int_q10    = []
mask_int_q90    = []
mask_int_min    = []
mask_int_max    = []

for patient_i in os.listdir(f'{path_var}'):
    
    print(f'PATIENT {patient_i}')
    
    for slice_i in os.listdir(f'{path_var}{patient_i}'):
        
        var_img = cv.imread(f'{path_var}{patient_i}.png', cv.IMREAD_GRAYSCALE)
        mask=1
        segments = slic(var_img, n_segments=10, compactness=3, start_label=1, mask=mask)
        
        if os.path.isfile(f'{path_processed}mask_prostate/{patient_i}/{filename_slice}.png') & \
           os.path.isfile(f'{path_processed}mask_cancer/{patient_i}/{filename_slice}.png'): # check if mask exists for this cycle/slice
            
            contrast_img = cv.imread(f'{path_processed}full_normalized/{patient_i}/{cycle_slice_i}')
            prostate_img = cv.imread(f'{path_processed}mask_prostate/{patient_i}/{filename_slice}.png')
            cancer_img = cv.imread(f'{path_processed}mask_cancer/{patient_i}/{filename_slice}.png')



img_type        = []
patient_id      = []
cycle_id        = []
slice_id        = []
mask_int_mean   = []



norm_img = cv.imread(f'{path_norm}50/15-12.png')
cv.imshow('', norm_img)
var_img = cv.imread(f'{path_var}50/norm 12.png')
prostate_mask = cv.imread(f'{path_maskP}/50/12.png', cv.IMREAD_GRAYSCALE)
segments = slic(var_img, n_segments=10, compactness=5, start_label=1, mask=prostate_mask)
cv.imshow('', mark_boundaries(var_img, segments, color=(0,0,1)))

cancer_mask = cv.imread(f'{path_maskC}/50/12.png', cv.IMREAD_GRAYSCALE)
x = segments[np.where((prostate_mask == 255) & (cancer_mask == 255))]
np.unique(x, return_counts=True)
# 1 - 351, 2 - 3, 3 - 644, 4 - 825, 7 - 238
np.unique(segments, return_counts=True)
# 0 - 253866, 1 - 458, 2 - 1151, 3 - 785, 4 - 1086, 5 - 1031, 6 - 1132, 7 - 513, 8 - 810, 9 - 861, 10 - 451
# % overlap: 1 - 77%, 2 - 1%, 3 - 82%, 4 - 76%, 7 - 46%

for cycle_slice_i in os.listdir(f'{path_norm}50'):
        
    filename_cycle = os.path.splitext(cycle_slice_i)[0].split('-')[0]
    filename_slice = os.path.splitext(cycle_slice_i)[0].split('-')[1]
        
    if filename_slice == '12':
        
        print(f'{cycle_slice_i}')
        
        slice_i = cv.imread(f'{path_norm}50/{cycle_slice_i}')
        
        for seg_i in range(1,10+1):
                              
            if seg_i == 1:
                img_type.append(0.77)
            elif seg_i == 2:
                img_type.append(0.01)
            elif seg_i == 3:
                img_type.append(0.82)
            elif seg_i == 4:
                img_type.append(0.76)
            elif seg_i == 7:
                img_type.append(0.46)
            else:
                img_type.append(0)
                
            patient_id.append(int(50))
            cycle_id.append(int(filename_cycle))
            slice_id.append(int(12))
            
            selected_region = slice_i[np.where(segments == seg_i)]
            mask_int_mean.append(np.mean(selected_region))

y = pd.DataFrame({
    'img_type':     img_type,
    'patient_id':     patient_id,
    'cycle_id':   cycle_id,
    'slice_id':   slice_id,
    'mask_int_mean':   mask_int_mean
    })

fig = plt.figure(dpi=150, figsize=(12,5))
fig.subplots_adjust(hspace=0.3, wspace=0.3)
ax = fig.add_subplot(1, 3, 3)
sns.lineplot(data=y.query(f'patient_id == 50'),
             x="cycle_id",
             y="mask_int_mean",
             hue="img_type",
             units="slice_id",
             estimator=None, lw=2).set_title(f'patient_id=50 int_mean')
ax = fig.add_subplot(1, 3, 1)
plt.imshow(norm_img)
plt.title("t=15 slice=12")
ax = fig.add_subplot(1, 3, 2)
plt.imshow(mark_boundaries(var_img, segments, color=(1,0,0)))
plt.title("std map + SLIC")
plt.savefig(f'{path_processed}test1.png', dpi=150)
plt.close()


seg_id = np.unique(segments)

intavg = []
for seg_i in seg_id:
    intavg.append(np.mean(img[np.where(segments == seg_i)]))
    img[np.where(segments == seg_i)] = np.mean(img[np.where(segments == seg_i)])
plt.bar(x=seg_id, height=intavg)

img3 = np.floor(255 / segments)
cv.imshow('', img)

img2 = img
img2[np.where(segments == 42)] = 255
img2[np.where(segments == 14)] = 230
img2[np.where(segments == 63)] = 210
img2[np.where(segments == 38)] = 190
img2[np.where(segments == 50)] = 170
cv.imshow('', img2)




### HOG IMPLEMENTATION

from skimage.feature import hog

img = cv.imread('C:/Users/Roman/Desktop/bllanos_20161006_spheres.png')
img = cv.imread(f'{path_norm}50/12-12.png')
cv.imshow('', img)

fd, hog_image = hog(img, orientations=4, pixels_per_cell=(16, 16), visualize=True)
cv.imshow('', hog_image)

