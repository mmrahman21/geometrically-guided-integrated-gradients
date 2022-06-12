'''

'''

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import seaborn as sns
import pandas as pd
from numpy import mean, median
import  numpy as np
import os
import sys
from statannot import add_stat_annotation

from myscripts.rar_plot_helper import create_per_method_result
# sns.set(style="darkgrid", palette="pastel", font_scale=1.0)
sns.set(style="whitegrid", font_scale=1.9)
# sns.set_theme()


import numpy as np
import pandas as pd
import os

Dataset = {0: 'oldtwocls', 1: 'newtwocls', 2: 'threecls', 3: 'multiwaytwocls', 4: 'uniformtwocls', 5: 'topdownposneg', 6: 'topdownnegneg', 7: 'topdownzero'}

data = 0

g = 1.0 

dir = "NPT"
mainDir = "MyStride1Dir"

prefix = "method_research_"+Dataset[data]+"_NPT_ud_rseam_"+str(g)+"_"

filenamepath = "../wandb"
filenamepath = os.path.join(filenamepath, 'Sequence_Based_Models', mainDir, dir, 'Saliency')


out_0 = filenamepath+"/"+prefix+"out_0_robyn_metric.npz"
out_1 = filenamepath+"/"+prefix+"out_1_robyn_metric.npz"

npzfile0 = np.load(out_0)
npzfile1 = np.load(out_1)
print('npzfile.files: {}'.format(npzfile0.files)) 

out_dict0 = {}
out_dict1 = {}

for k in npzfile0.files:
    out_dict1[k+".1"] = npzfile1[k].flatten()
    print('out_dict1[{}]: {}'.format(k+".1", out_dict1[k+".1"].shape))
    out_dict0[k+".0"] = npzfile0[k].flatten()
    print('ou_dict0[{}]: {}'.format(k+".0", out_dict0[k+".0"].shape))
    
out0_df = pd.DataFrame.from_dict(out_dict0)
print(out0_df)
out1_df = pd.DataFrame.from_dict(out_dict1)
print(out1_df)


# If 1x4 plot, then follow this code
######################################
color_palette= ['#7b3294','#c2a5cf','#a6dba0','#008837']
color_palette2 = ['#d7191c','#fdae61','#abd9e9','#2c7bb6']
color_palette3 = ['#ca0020','#f4a582','#92c5de','#0571b0']

div_4_class = ['#e66101','#fdb863','#b2abd2','#5e3c99']
div_4_class_2 = ['#e66101','#fdb863','#b2abd2','#5e3c99']

seven_class = ['#b2182b','#ef8a62','#fddbc7','#f7f7f7','#d1e5f0','#67a9cf','#2166ac']

eight_class = ['#d73027','#f46d43','#fdae61','#fee090','#e0f3f8','#abd9e9','#74add1','#4575b4']

sns.set_palette(palette=seven_class)


f, axes = plt.subplots(1, 3, figsize=(25,5))

g1 = sns.boxplot(x="SPC", y="AUC",linewidth=1, width=0.6,
            hue="Method",
            data=dataframes_collection[0],ax=axes[0])

g2 = sns.boxplot(x="SPC", y="AUC",linewidth=1, width=0.72,
            hue="Method",
            data=dataframes_collection[1],ax=axes[1])


g3 = sns.boxplot(x="SPC", y="AUC",linewidth=1, width=0.72,
            hue="Method",
            data=dataframes_collection[2],ax=axes[2])

hue = "Method"

boxes = []
data = 0  # FBIRN
subjects_per_group = [int(item) for item in dataset_columns[Dataset[data]]]

for g in range(len(subjects_per_group)):
    item = ((subjects_per_group[g], "MILC_UFPT"), (subjects_per_group[g], "MILC_NPT"))
    boxes.append(item)

add_stat_annotation(g1, data=dataframes_collection[0], x="SPC", y="AUC", hue=hue,
                    box_pairs=boxes, test='Wilcoxon', text_format='star', loc='inside', verbose=2)

boxes = []
data = 2  # OASIS
subjects_per_group = [int(item) for item in dataset_columns[Dataset[data]]]

for g in range(len(subjects_per_group)):
    item = ((subjects_per_group[g], "MILC_UFPT"), (subjects_per_group[g], "MILC_NPT"))
    boxes.append(item)

add_stat_annotation(g2, data=dataframes_collection[1], x="SPC", y="AUC", hue=hue,
                    box_pairs=boxes, test='Wilcoxon', text_format='star', loc='inside', verbose=2)

boxes = []
data = 3   # ABIDE
subjects_per_group = [int(item) for item in dataset_columns[Dataset[data]]]

for g in range(len(subjects_per_group)):
    item = ((subjects_per_group[g], "MILC_UFPT"), (subjects_per_group[g], "MILC_NPT"))
    boxes.append(item)

add_stat_annotation(g3, data=dataframes_collection[2], x="SPC", y="AUC", hue=hue,
                    box_pairs=boxes, test='Wilcoxon', text_format='star', loc='inside', verbose=2)




# g4 = sns.boxplot(x="SPC", y="AUC",linewidth=2, width=0.75,
#             hue="Method",
#             data=dataframes_collection[3],ax=axes[3])


axes[0].get_legend().remove()
axes[1].get_legend().remove()
# axes[2].get_legend().remove()

axes[0].set(ylim=(0.2, 1.0))
axes[1].set(ylim=(0.2, 1.0))
axes[2].set(ylim=(0.2, 1.0))
# axes[3].set(ylim=(0.3, 0.90))

g1.set(yticks=np.arange(0.2, 1.01,0.1))
g2.set(yticks=np.arange(0.2, 1.01,0.1))
g3.set(yticks=np.arange(0.2, 1.01,0.1))
# g4.set(yticks=np.arange(0.3,0.91,0.1))

g1.set_title('FBIRN')
# g2.set_title('COBRE')
g2.set_title('OASIS')
g3.set_title('ABIDE')
# axes[0].set_xlabel('')
# axes[1].set_xlabel('')

axes[1].set_ylabel('')
axes[2].set_ylabel('')
# axes[3].set_ylabel('')

axes[1].yaxis.set_ticklabels([])
# axes[3].yaxis.set_ticklabels([])
axes[2].yaxis.set_ticklabels([])
handles, labels = axes[0].get_legend_handles_labels()

axes[2].legend(handles[:], labels[:], prop={'size': 15}, loc=2, title="Method", borderaxespad=0.)
# sns.despine(offset=1, trim=True)

f.tight_layout()
plt.subplots_adjust(wspace=0.01, hspace=0)

plt.show()


# f.savefig('../wandb/Sequence_Based_Models/'+'All_DATA_All_spc_SML_MILC_multi_fold_reviesd_width_height.svg', transparent=True, bbox_inches='tight', pad_inches=0)
# f.savefig('../wandb/Sequence_Based_Models/'+'All_DATA_All_spc_SML_MILC_multi_fold_plot.png', format='png', dpi=600)