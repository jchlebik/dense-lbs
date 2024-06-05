# Path: tensorboard_reader.py

import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import io

import pandas as pd

from tbparse import SummaryReader

# cwd = os.getcwd()
# file_paths = {
#     os.path.join(os.getcwd(), "results/12s8c/2024-03-29_01-15-53/tensorboard/events.out.tfevents.1711671419.acn04.karolina.it4i.cz.77055.0.v2"),
    
# }
        


#file_path = (os.path.join(os.getcwd(), "/home/xchleb07/dev/proj/karolina/dlbs/2024-05-14_16-08-07/tensorboard/events.out.tfevents.1715695687.acn52.karolina.it4i.cz.68435.0.v2"))
file_path = (os.path.join(os.getcwd(), "/home/xchleb07/dev/proj/karolina/corrector/2024-05-14_16-23-26/tensorboard/events.out.tfevents.1715696606.acn65.karolina.it4i.cz.74946.0.v2"))

# log_dir = file_path
reader = SummaryReader(file_path, pivot=True)
df = reader.tensors
#['step', 'Loss/train', 'Loss/validation', 'Density', 'Source', 'Speed of sound', 'Field', 'Predicted Field', ]

hp = {k: v[0] for k, v in reader.hparams.to_dict(orient='list').items()}

nan_mask = ~df['Density'].isnull()
sos_it = df.loc[nan_mask, 'Speed of sound']
density_it = df.loc[nan_mask, 'Density']
field_it = df.loc[nan_mask, 'True Field']
pred_field_it = df.loc[nan_mask, 'Predicted Field']
diff_it = df.loc[nan_mask, 'Relative Difference %']


tb_index = 8434

i = np.where(diff_it.index.values == tb_index)[0][0]

diff = diff_it.values[i]
diff_raster = plt.imread(io.BytesIO(diff[2]))
plt.imsave('diff.png', diff_raster)

#diff_it = reversed(diff_it)






for (sos, rho, field, pred, diff) in zip(sos_it, density_it, field_it, pred_field_it, diff_it):
    sos_raster = plt.imread(io.BytesIO(sos[2]))
    plt.imsave('sos.png', sos_raster)
    
    density_raster = plt.imread(io.BytesIO(rho[2]))
    plt.imsave('density.png', density_raster)
        
    field_raster = plt.imread(io.BytesIO(field[2]))
    plt.imsave('field.png', field_raster)
    
    pred_field_raster = plt.imread(io.BytesIO(pred[2]))
    plt.imsave('pred_field.png', pred_field_raster)

    diff_raster = plt.imread(io.BytesIO(diff[2]))
    plt.imsave('diff.png', diff_raster)
    

