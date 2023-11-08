import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.interpolate import griddata

#process the csv results
sns.set_theme(style="whitegrid")
pesq_bool = False
df_result = pd.read_csv("/dsi/scratch/from_netapp/users/mordehay/Results/"
                        "Separation_data_wham_partial_overlap_ModoluDial_Vad099_decay095_inputActivity_"
                        "AttentionBeforeSum_2rSkippLN_Test2/results_information.csv") #todo- general path
df_information = pd.read_csv("/dsi/gannot-lab/datasets/mordehay/data_wham_partial_overlap/test/csv_files/"
                             "with_wham_noise_res_more_informative.csv")
df_information = df_information[5:] #todo- our bug
df_information = df_information.set_index(df_result.index)
df_result = df_result[df_information["speaker1_id"] != df_information["speaker2_id"]]
df_information = df_information[df_information["speaker1_id"] != df_information["speaker2_id"]]


save_test_path = "/dsi/scratch/from_netapp/users/mordehay/Results/" \
                 "Separation_data_wham_partial_overlap_ModoluDial_Vad099_decay095_inputActivity_" \
                 "AttentionBeforeSum_2rSkippLN_Test2/"
save_test_path = save_test_path + "Postprocessing/"
Path(f"{save_test_path}").mkdir(parents=True, exist_ok=True)
num_samples_show = 50
##scatter of reverb and snr, si_sdr. This scatter is divided by gender

"""
snr_same = df_result[df_information["same_gender"] == 1]["snr"]
reverb_same = df_result[df_information["same_gender"] == 1]["reverb"]
si_sdr_same = df_result[df_information["same_gender"] == 1]["si_sdr"]"""


snr_same = df_result[df_information["same_gender"] == 1]["snr"][:num_samples_show]
reverb_same = df_result[df_information["same_gender"] == 1]["reverb"][:num_samples_show]
si_sdr_same = df_result[df_information["same_gender"] == 1]["si_sdr"][:num_samples_show]

snr_diff = df_result[df_information["same_gender"] == 0]["snr"][:num_samples_show]
reverb_diff = df_result[df_information["same_gender"] == 0]["reverb"][:num_samples_show]
si_sdr_diff = df_result[df_information["same_gender"] == 0]["si_sdr"][:num_samples_show]

fig, ax = plt.subplots()
ax.scatter(snr_same, reverb_same, c=si_sdr_same,
            linewidths = 2,
            marker ="s",
            label="same_gender",
            s = 50)
 
g = ax.scatter(snr_diff, reverb_diff, c=si_sdr_diff,
            linewidths = 2,
            marker ="^",
            label="diff_gender",
            s = 50)


"""fig, ax = plt.subplots()
g = ax.scatter(x = df_result["reverb"],
            y = np.array(df_result["snr"]),
            c = df_result["si_sdr"],
            cmap = "magma")"""


cbar =fig.colorbar(g)
cbar.set_label('SI-SDR[dB]', rotation=270)
plt.xlabel("T$_{60}$[sec]")
ax.set_xlabel("SNR[dB]")
#save_test_path= '/home/lab/renana/PycharmProjects/AV_rtf_separation/'
plt.savefig(f"{save_test_path}Plot_reverb_snr_sisdr_sameAnddiffGender.png")
plt.show()
plt.close()

interpolate_bool = False
if interpolate_bool:
    # Example data points
    snr = df_result["snr"]
    reverb = df_result["reverb"]
    sisdr = df_result["si_sdr"]

    # Define a grid of x and y coordinates
    snri = np.linspace(min(snr), max(snr), 5000)
    reverbi = np.linspace(min(reverb), max(reverb), 5000)
    X, Y = np.meshgrid(snri, reverbi)

    # Interpolate the z values for the grid points
    Z = griddata((snr, reverb), sisdr, (X, Y), method='cubic')

    # Plot the interpolated image
    plt.imshow(Z, extent=(min(snr), max(snr), min(reverb), max(reverb)), origin='lower', cmap='jet', aspect='auto', vmin=-4, vmax=15)
    plt.colorbar()
    plt.savefig(f"{save_test_path}Plot_reverb_snr_sisdr_sameAnddiffGender_continous.png")
    plt.close()

#df_result["snr"].apply(lambda x: pd.qcut(x, 3, labels=['low', 'medium', 'high']))

df_result["snr_cat"] = pd.qcut(df_result["snr"], 3, labels=['0-5', '5-10', '10-15'])
df_result["rt60_cat"] = pd.qcut(df_result["reverb"], 3, labels=['0.2-0.33', '0.33-0.46', '0.46-0.6'])
#df_result["drr"] = df_result["drr"]
df_result["speaker0_gender"] = df_information["speaker0_gender"]
df_result["speaker1_gender"] = df_information["speaker1_gender"]
df_result["gender"] = df_result[df_information["same_gender"] == 1]["speaker0_gender"]
df_result["gender"][df_information["same_gender"] == 0] = "Male-female"

df_result["gender"] [df_result["gender"] == 0]= "Female-female"
df_result["gender"] [df_result["gender"] == 1] = "Male-male"

df_result["drr"] = 0
df_result["drr"][(df_information["speaker0_far_drr"] == 0) & (df_information["speaker1_far_drr"] == 0)] = 'Close'
df_result["drr"][(df_information["speaker0_far_drr"] == 1) & (df_information["speaker1_far_drr"] == 1)] = 'Far'
df_result["drr"][((df_information["speaker0_far_drr"] == 0) & (df_information["speaker1_far_drr"] == 1)) | ((df_information["speaker0_far_drr"] == 1) & (df_information["speaker1_far_drr"] == 0))] = 'Close&far'




box_plot = sns.boxplot(x="snr_cat", y="si_sdri", data=df_result, showmeans=True, meanprops={"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"5"})
plt.ylabel("SI-SDRi[dB]")
plt.xlabel("SNR[dB]")
medians = df_result.groupby(['snr_cat'])['si_sdri'].median()
vertical_offset = medians * 0.05 # offset from median for display
labels = box_plot.get_xticklabels()
for xtick in box_plot.get_xticks():
    label = labels[xtick].get_text()
    box_plot.text(xtick, medians[label] + vertical_offset[label], round(medians[label], 2), 
            horizontalalignment='center',size='x-small',color='w',weight='semibold')
plt.savefig(f"{save_test_path}sisdri_vs_snr_cat.png")
plt.show()
plt.close()

box_plot = sns.boxplot(x="snr_cat", y="si_sdr", data=df_result, showmeans=True, meanprops={"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"5"})
medians = df_result.groupby(['snr_cat'])['si_sdr'].median()
vertical_offset = medians * 0.05 # offset from median for display
labels = box_plot.get_xticklabels()
for xtick in box_plot.get_xticks():
    label = labels[xtick].get_text()
    box_plot.text(xtick,medians[label] + vertical_offset[label], round(medians[label], 2), 
            horizontalalignment='center',size='x-small',color='w',weight='semibold')
plt.ylabel("SI-SDR[dB]")
plt.xlabel("SNR[dB]")
plt.savefig(f"{save_test_path}sisdr_vs_snr_cat.png")
plt.show()
plt.close()

box_plot = sns.boxplot(x="rt60_cat", y="si_sdri", data=df_result, showmeans=True, meanprops={"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"5"})
medians = df_result.groupby(['rt60_cat'])['si_sdri'].median()
vertical_offset = medians * 0.05 # offset from median for display
labels = box_plot.get_xticklabels()
for xtick in box_plot.get_xticks():
    label = labels[xtick].get_text()
    box_plot.text(xtick, medians[label] + vertical_offset[label], round(medians[label], 2), 
            horizontalalignment='center',size='x-small',color='w',weight='semibold')
plt.ylabel("SI-SDRi[dB]")
plt.xlabel("T$_{60}$[sec]")
plt.savefig(f"{save_test_path}sisdri_vs_rt60_cat.png")
plt.show()
plt.close()

box_plot = sns.boxplot(x="rt60_cat", y="si_sdr", data=df_result, showmeans=True, meanprops={"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"5"})
medians = df_result.groupby(['rt60_cat'])['si_sdr'].median()
vertical_offset = medians * 0.05 # offset from median for display
labels = box_plot.get_xticklabels()
for xtick in box_plot.get_xticks():
    label = labels[xtick].get_text()
    box_plot.text(xtick, medians[label] + vertical_offset[label], round(medians[label], 2), 
            horizontalalignment='center',size='x-small',color='w',weight='semibold')
plt.ylabel("SI-SDR[dB]")
plt.xlabel("T$_{60}$[sec]")
plt.savefig(f"{save_test_path}sisdr_vs_rt60_cat.png")
plt.show()
plt.close()

"""fig, ax = plt.subplots()
g = ax.scatter(x = df["reverb"], 
            y = -np.array(df["snr"]),
            c = df["si_sdr"],
            cmap = "magma")"""

"""fig.colorbar(g)
ax.set_ylabel("reverbs[ms]")
ax.set_xlabel("snr[dB]")
ax.legend()
ax.grid(True)
plt.savefig(f"{save_test_path}Plot_reverb_snr_sisdr_sameAnddiffGender.png")
plt.close()"""

box_plot = sns.boxplot(x="drr", y="si_sdri", data=df_result, showmeans=True, meanprops={"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"5"})
medians = df_result.groupby(['drr'])['si_sdri'].median()
vertical_offset = medians * 0.05 # offset from median for display
labels = box_plot.get_xticklabels()
for xtick in box_plot.get_xticks():
    label = labels[xtick].get_text()
    box_plot.text(xtick, medians[label] + vertical_offset[label] , round(medians[label], 2), 
            horizontalalignment='center',size='x-small',color='w',weight='semibold')
plt.ylabel("SI-SDRi[dB]")
plt.xlabel("Speakers' to mic distance")
plt.savefig(f"{save_test_path}sisdri_vs_drr.png")
plt.close()
plt.show()

box_plot = sns.boxplot(x="drr", y="si_sdr", data=df_result, showmeans=True, meanprops={"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"5"})
medians = df_result.groupby(['drr'])['si_sdr'].median()
vertical_offset = medians * 0.05 # offset from median for display
labels = box_plot.get_xticklabels()
for xtick in box_plot.get_xticks():
    label = labels[xtick].get_text()
    box_plot.text(xtick, medians[label] + vertical_offset[label] , round(medians[label], 2), 
            horizontalalignment='center',size='x-small',color='w',weight='semibold')
plt.ylabel("SI-SDR[dB]")
plt.xlabel("Speakers' to mic distance")
plt.savefig(f"{save_test_path}sisdr_vs_drr.png")
plt.show()
plt.close()

#sns.boxplot(x="same_gender", y="si_sdri", data=df_result)
#plt.show()

box_plot = sns.boxplot(x="gender", y="si_sdri", data=df_result, showmeans=True, meanprops={"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"5"})
medians = df_result.groupby(['gender'])['si_sdri'].median()
vertical_offset = medians * 0.05 # offset from median for display
labels = box_plot.get_xticklabels()
for xtick in box_plot.get_xticks():
    label = labels[xtick].get_text()
    box_plot.text(xtick, medians[label] + vertical_offset[label], round(medians[label], 2), 
            horizontalalignment='center',size='x-small',color='w',weight='semibold')
plt.ylabel("SI-SDRi[dB]")
plt.xlabel("")
plt.savefig(f"{save_test_path}sisdri_vs_gender_boxplot.png")
plt.show()
plt.close()

box_plot = sns.boxplot(x="gender", y="si_sdr", data=df_result, showmeans=True, meanprops={"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"5"})
medians = df_result.groupby(['gender'])['si_sdr'].median()
vertical_offset = medians * 0.05 # offset from median for display
labels = box_plot.get_xticklabels()
for xtick in box_plot.get_xticks():
    label = labels[xtick].get_text()
    box_plot.text(xtick, medians[label] + vertical_offset[label], round(medians[label], 2), 
            horizontalalignment='center',size='x-small',color='w',weight='semibold')
plt.ylabel("SI-SDR[dB]")
plt.xlabel("")
plt.savefig(f"{save_test_path}sisdr_vs_gender_boxplot.png")
plt.show()
plt.close()

sns.violinplot(x="gender", y="si_sdr", data=df_result)
plt.savefig(f"{save_test_path}sisdr_vs_gender_violinplot.png")
plt.show()
plt.close()

sns.histplot(data=df_result, x="si_sdri", kde=True)
plt.savefig(f"{save_test_path}sisdri_kde.png")
plt.show()
plt.close()


sns.histplot(data=df_result, x="si_sdr_start", kde=True)
sns.histplot(data=df_result, x="si_sdr", kde=True, color="purple")
label_input = f'si_sdr_input = {round(df_result["si_sdr_start"].mean(), 2)}'
label_output = f'si_sdr_output = {round(df_result["si_sdr"].mean(), 2)}'
plt.legend(labels=[label_input, label_output])
plt.xlabel("SI-SDR[dB]")
plt.ylabel("Count")
plt.savefig(f"{save_test_path}sisdrStart_final_kde.png")
plt.show()
plt.close()


"""sns.histplot(data=df_result, x="si_sdr_start", kde=True)
sns.histplot(data=df_result, x="online_sisdr", kde=True, color="purple")
label_input = f'si_sdr_input = {round(df_result["si_sdr_start"].mean(), 2)}'
label_output = f'si_sdr_output = {round(df_result["online_sisdr"].mean(), 2)}'
plt.legend(labels=[label_input, label_output])
plt.xlabel("Online SI-SDR[dB]")
plt.ylabel("Count")
plt.savefig(f"{save_test_path}OnlinesisdrStart_final_kde.png")
plt.show()
plt.close()"""
#################### Pesq Vs Snr, RT60, gender
if pesq_bool:
    ## Gender
    df_result["pesq"] = df_result[["pesq_spk0", "pesq_spk1"]].mean(axis=1)

    box_plot = sns.boxplot(x="gender", y="pesq", data=df_result, showmeans=True, meanprops={"marker":"o",
                        "markerfacecolor":"white", 
                        "markeredgecolor":"black",
                        "markersize":"5"})
    medians = df_result.groupby(['gender'])['pesq'].median()
    vertical_offset = medians * 0.001 # offset from median for display
    labels = box_plot.get_xticklabels()
    for xtick in box_plot.get_xticks():
        label = labels[xtick].get_text()
        box_plot.text(xtick, medians[label] + vertical_offset[label], round(medians[label], 2), 
                horizontalalignment='center',size='x-small',color='w',weight='semibold')
    plt.savefig(f"{save_test_path}pesq_vs_gender_boxplot.png")
    plt.show()
    plt.close()

    ## RT60
    box_plot = sns.boxplot(x="rt60_cat", y="pesq", data=df_result, showmeans=True, meanprops={"marker":"o",
                        "markerfacecolor":"white", 
                        "markeredgecolor":"black",
                        "markersize":"5"})
    medians = df_result.groupby(['rt60_cat'])['pesq'].median()
    vertical_offset = medians * 0.001 # offset from median for display
    labels = box_plot.get_xticklabels()
    for xtick in box_plot.get_xticks():
        label = labels[xtick].get_text()
        box_plot.text(xtick, medians[label] + vertical_offset[label], round(medians[label], 2), 
                horizontalalignment='center',size='x-small',color='w',weight='semibold')
    plt.savefig(f"{save_test_path}pesq_vs_rt60_cat_boxplot.png")
    plt.show()
    plt.close()

    ## SNR
    box_plot = sns.boxplot(x="snr_cat", y="pesq", data=df_result, showmeans=True, meanprops={"marker":"o",
                        "markerfacecolor":"white", 
                        "markeredgecolor":"black",
                        "markersize":"5"})
    medians = df_result.groupby(['snr_cat'])['pesq'].median()
    vertical_offset = medians * 0.001 # offset from median for display
    labels = box_plot.get_xticklabels()
    for xtick in box_plot.get_xticks():
        label = labels[xtick].get_text()
        box_plot.text(xtick, medians[label] + vertical_offset[label], round(medians[label], 2), 
                horizontalalignment='center',size='x-small',color='w',weight='semibold')
    plt.savefig(f"{save_test_path}pesq_vs_snr_cat_boxplot.png")
    plt.show()
    plt.close()

    ## DRR
    box_plot = sns.boxplot(x="drr", y="pesq", data=df_result, showmeans=True, meanprops={"marker":"o",
                        "markerfacecolor":"white", 
                        "markeredgecolor":"black",
                        "markersize":"5"})
    medians = df_result.groupby(['drr'])['pesq'].median()
    vertical_offset = medians * 0.001 # offset from median for display
    labels = box_plot.get_xticklabels()
    for xtick in box_plot.get_xticks():
        label = labels[xtick].get_text()
        box_plot.text(xtick, medians[label] + vertical_offset[label], round(medians[label], 2), 
                horizontalalignment='center',size='x-small',color='w',weight='semibold')
    plt.savefig(f"{save_test_path}pesq_vs_drr_boxplot.png")
    plt.show()
    plt.close()



#################### Stoi Vs Snr, RT60, gender
## Gender
box_plot = sns.boxplot(x="gender", y="stoi", data=df_result, showmeans=True, meanprops={"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"5"})
medians = df_result.groupby(['gender'])['stoi'].median()
vertical_offset = medians * 0.001 # offset from median for display
#print(medians)
#print(vertical_offset)
labels = box_plot.get_xticklabels()
for xtick in box_plot.get_xticks():
    label = labels[xtick].get_text()
    #print(label)
    #print(medians[label] + vertical_offset[label])
    box_plot.text(xtick, medians[label] + vertical_offset[label], round(medians[label], 2), 
            horizontalalignment='center',size='x-small',color='w',weight='semibold')
plt.savefig(f"{save_test_path}stoi_vs_gender_boxplot.png")
plt.show()
plt.close()

## RT60
box_plot = sns.boxplot(x="rt60_cat", y="stoi", data=df_result, showmeans=True, meanprops={"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"5"})
medians = df_result.groupby(['rt60_cat'])['stoi'].median()
vertical_offset = medians * 0.001 # offset from median for display
labels = box_plot.get_xticklabels()
for xtick in box_plot.get_xticks():
    label = labels[xtick].get_text()
    box_plot.text(xtick, medians[label] + vertical_offset[label], round(medians[label], 2), 
            horizontalalignment='center',size='x-small',color='w',weight='semibold')
plt.savefig(f"{save_test_path}stoi_vs_rt60_cat_boxplot.png")
plt.show()
plt.close()

## SNR
box_plot = sns.boxplot(x="snr_cat", y="stoi", data=df_result, showmeans=True, meanprops={"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"5"})
medians = df_result.groupby(['snr_cat'])['stoi'].median()
vertical_offset = medians * 0.001 # offset from median for display
labels = box_plot.get_xticklabels()
for xtick in box_plot.get_xticks():
    label = labels[xtick].get_text()
    box_plot.text(xtick, medians[label] + vertical_offset[label], round(medians[label], 2), 
            horizontalalignment='center',size='x-small',color='w',weight='semibold')
plt.savefig(f"{save_test_path}stoi_vs_snr_cat_boxplot.png")
plt.show()
plt.close()

## DRR
box_plot = sns.boxplot(x="drr", y="stoi", data=df_result, showmeans=True, meanprops={"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"5"})
medians = df_result.groupby(['drr'])['stoi'].median()
vertical_offset = medians * 0.001 # offset from median for display
labels = box_plot.get_xticklabels()
for xtick in box_plot.get_xticks():
    label = labels[xtick].get_text()
    box_plot.text(xtick, medians[label] + vertical_offset[label], round(medians[label], 2), 
            horizontalalignment='center',size='x-small',color='w',weight='semibold')
plt.savefig(f"{save_test_path}stoi_vs_drr_boxplot.png")
plt.show()
plt.close()


########################
sns.histplot(data=df_result, x="initial_stoi", kde=True)
sns.histplot(data=df_result, x="stoi", kde=True, color="purple")
plt.legend(labels=[f"stoi_input = {round(df_result['initial_stoi'].mean(), 2)}",f"stoi_output = {round(df_result['stoi'].mean(), 2)}"])
plt.xlabel("Stoi")
plt.ylabel("Frequency")
plt.savefig(f"{save_test_path}stoiStart_final_kde.png")
plt.show()
plt.close()

if pesq_bool:
    sns.histplot(data=df_result, x="pesq_initial", kde=True)
    sns.histplot(data=df_result, x="pesq", kde=True, color="purple")
    plt.legend(labels=[f"pesq_input = {round(df_result['pesq_initial'].mean(), 2)}",f"pesq_output = {round(df_result['pesq'].mean(), 2)}"])
    plt.xlabel("Pesq")
    plt.ylabel("Frequency")
    plt.savefig(f"{save_test_path}pesqStart_final_kde.png")
    plt.show()
    plt.close()
##scatter of reverb and snr, si_sdr. This scatter is divided by drr. 
#There are three cases: 1. The DRR of two speakers is greater than the critical distnace.
#                       2. The DRR of one of the speakers is greater than the critical distnace.
#                       3. The DRR of two speakers is smaller than the critical distnace.

two_far_drr = df_information["speaker0_far_drr"] & df_information["speaker1_far_drr"] #for case 1
one_far_drr = df_information["speaker0_far_drr"] ^ df_information["speaker1_far_drr"] #for case 2
zero_far_drr = 1 - (df_information["speaker0_far_drr"] | df_information["speaker1_far_drr"]) #for case 3


##plot the confusion matrix
tp_0 = df_result["tp_0"]
fp_0 = df_result["fp_0"]
fn_0 = df_result["fn_0"]
tn_0 = df_result["tn_0"]

tp_1 = df_result["tp_1"]
fp_1 = df_result["fp_1"]
fn_1 = df_result["fn_1"]
tn_1 = df_result["tn_1"]

tp_0_mean = np.mean(df_result["tp_0"])
fp_0_mean = np.mean(df_result["fp_0"])
fn_0_mean = np.mean(df_result["fn_0"])
tn_0_mean = np.mean(df_result["tn_0"])

tp_1_mean = np.mean(df_result["tp_1"])
fp_1_mean = np.mean(df_result["fp_1"])
fn_1_mean = np.mean(df_result["fn_1"])
tn_1_mean = np.mean(df_result["tn_1"])

cf_matrix_0 = [[tn_0_mean, fp_0_mean], [fn_0_mean, tp_0_mean]]
cf_matrix_1 = [[tn_1_mean, fp_1_mean], [fn_1_mean, tp_1_mean]]

cf_matrix_0_norm = cf_matrix_0 / np.sum(cf_matrix_0, axis=-1, keepdims=True)
cf_matrix_1_norm = cf_matrix_1 / np.sum(cf_matrix_1, axis=-1, keepdims=True)

ax = sns.heatmap(cf_matrix_0, annot=True, cmap='Blues')

ax.set_title('Our VAD confusion matrix\n')
ax.set_xlabel('Predicted Values')
ax.set_ylabel('True Values')

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False - Non Active','True - Active'])
ax.yaxis.set_ticklabels(['False - Non Active','True - Active'])
plt.savefig(f"{save_test_path}ConfusionMatrix_spk0_OurVad.png")
plt.close()

ax = sns.heatmap(cf_matrix_1, annot=True, cmap='Blues')

ax.set_title('Our VAD confusion matrix\n')
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values ')

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False - Non Active','True - Active'])
ax.yaxis.set_ticklabels(['False - Non Active','True - Active'])
plt.savefig(f"{save_test_path}ConfusionMatrix_spk1_OurVad.png")
plt.close()



ax = sns.heatmap(cf_matrix_0_norm, annot=True, cmap='Blues')

ax.set_title('Our VAD confusion matrix\n')
ax.set_xlabel('Predicted Values')
ax.set_ylabel('True Values')

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False - Non Active','True - Active'])
ax.yaxis.set_ticklabels(['False - Non Active','True - Active'])
plt.savefig(f"{save_test_path}ConfusionMatrix_spk0_OurVad_Normalized.png")
plt.close()

ax = sns.heatmap(cf_matrix_1_norm, annot=True, cmap='Blues')

ax.set_title('Our VAD confusion matrix\n')
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values ')

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False - Non Active','True - Active'])
ax.yaxis.set_ticklabels(['False - Non Active','True - Active'])
plt.savefig(f"{save_test_path}ConfusionMatrix_spk1_OurVad_Normalized.png")
plt.close()


##plot the confusion matrix
"""tp_simple0 = df_result["tp_simple0"]
fp_simple0 = df_result["fp_simple0"]
fn_simple0 = df_result["fn_simple0"]
tn_simple0 = df_result["tn_simple0"]

tp_simple1 = df_result["tp_simple1"]
fp_simple1 = df_result["fp_simple1"]
fn_simple1 = df_result["fn_simple1"]
tn_simple1 = df_result["tn_simple1"]"""
print(df_result.columns.tolist())
tp_0_mean = np.sum(df_result["tp_simple0"])
fp_0_mean = np.sum(df_result["fp_simple0"])
fn_0_mean = np.sum(df_result["fn_simple0"])
tn_0_mean = np.sum(df_result["tn_simple0"])

tp_1_mean = np.sum(df_result[" tp_simple1"])
fp_1_mean = np.sum(df_result["fp_simple1"])
fn_1_mean = np.sum(df_result["fn_simple1"])
tn_1_mean = np.sum(df_result["tn_simple1"])

cf_matrix_0 = [[tn_0_mean, fp_0_mean], [fn_0_mean, tp_0_mean]]
cf_matrix_1 = [[tn_1_mean, fp_1_mean], [fn_1_mean, tp_1_mean]]


ax = sns.heatmap(cf_matrix_0, annot=True, cmap='Blues')

ax.set_title('Simple VAD confusion matrix\n')
ax.set_xlabel('Predicted Values')
ax.set_ylabel('True Values')

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False - Non Active','True - Active'])
ax.yaxis.set_ticklabels(['False - Non Active','True - Active'])
plt.savefig(f"{save_test_path}ConfusionMatrix_spk0_simpleVad_sum.png")
plt.close()

ax = sns.heatmap(cf_matrix_1, annot=True, cmap='Blues')

ax.set_title('Simple VAD confusion matrix\n')
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values ')

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False - Non Active','True - Active'])
ax.yaxis.set_ticklabels(['False - Non Active','True - Active'])
plt.savefig(f"{save_test_path}ConfusionMatrix_spk1_simpleVad_sum.png")
plt.close()


##NNormalize


cf_matrix_0 = [[tn_0_mean, fp_0_mean], [fn_0_mean, tp_0_mean]]
cf_matrix_1 = [[tn_1_mean, fp_1_mean], [fn_1_mean, tp_1_mean]]

print(cf_matrix_0)
print(cf_matrix_1)
cf_matrix_0 = cf_matrix_0 / np.sum(cf_matrix_0, axis=-1, keepdims=True)
cf_matrix_1 = cf_matrix_1 / np.sum(cf_matrix_1, axis=-1, keepdims=True)
ax = sns.heatmap(cf_matrix_0, annot=True, cmap='Blues')

ax.set_title('Simple VAD confusion matrix\n')
ax.set_xlabel('Predicted Values')
ax.set_ylabel('True Values')

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False - Non Active','True - Active'])
ax.yaxis.set_ticklabels(['False - Non Active','True - Active'])
plt.savefig(f"{save_test_path}ConfusionMatrix_spk0_simpleVad_sum_Normalized.png")
plt.close()

ax = sns.heatmap(cf_matrix_1, annot=True, cmap='Blues')

ax.set_title('Simple VAD confusion matrix\n')
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values ')

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False - Non Active','True - Active'])
ax.yaxis.set_ticklabels(['False - Non Active','True - Active'])
plt.savefig(f"{save_test_path}ConfusionMatrix_spk1_simpleVad_sum_Normalized.png")
plt.close()


