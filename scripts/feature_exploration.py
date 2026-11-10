import os
from pathlib import Path
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.integrate import simpson


def plot_age_dist(df, out_dir):
    ''' Plot age distribution, save in OUT directory'''

    plt.figure(figsize=(9,4))
    plt.hist(df["Quantized Age (years)"], bins = np.arange(0,18,1))
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.title("Participant age distribution")
    plt.savefig(out_dir/"age_dist.pdf")

    return



def extract_band_power(signal, stage_intervals, sfreq = 64, window_s=60, step_s=30):
    '''Band power: Calculate power spectrum of a signal and then intergrate over each band'''
    bands = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 12), "beta": (12, 30)}
    
    nyquist = sfreq / 2  # for gamma, when changing to 64 hz nyquist does not allow for gamma 
    if nyquist > 45:
        bands["gamma"] = (30, 45)

    upper_frequency = max(high for low, high in bands.values())

    window_n = round(window_s * sfreq)
    step_n = round(step_s * sfreq)
    rows = []

    for stage, (start_s, stop_s) in stage_intervals.items():
        start_n = round(start_s * sfreq)
        stop_n = min(round(stop_s * sfreq), len(signal))

        for start in range(start_n, stop_n - window_n + 1, step_n):
            segment = signal[start:start + window_n]

            frequencies, psd = welch(segment, fs=sfreq, window="hann", nperseg=4*sfreq,
                                     noverlap=(2*sfreq), scaling="density")

            def power_between(low, high):
                inside = ((frequencies>low) & (frequencies<high))
                f = np.concatenate(([low], frequencies[inside], [high]))
                p = np.interp(f, frequencies, psd)
                return simpson(p, x=f)
            
            total_pwr = power_between(0.5,upper_frequency)

            row = {"stage": stage, "start_s": start / sfreq}
            for band, (low, high) in bands.items():
                absolute = power_between(low, high)
                row[f"{band}_absolute"] = absolute
                row[f"{band}_relative"] = absolute / total_pwr

            rows.append(row)
            
    return pd.DataFrame(rows)


def plot_median_band_pwr(annotations, out_dir, win_size=30, step_size=15):
    ''' plotting figures of median relative band power for few participants from each age_group. '''

    file_paths = [path for path, file_name, age_text in annotations]
    save_dir = out_dir / "relative_band_pwr" /f"win{win_size}_step{step_size}" /f"{len(annotations)/2}per_age_gr"
    os.makedirs(save_dir, exist_ok=True)

    montage = ['FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2','T3-C3', 'C3-CZ',
                'CZ-C4', 'C4-T4','FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1','FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',]
    bands = ["delta", "theta", "alpha", "beta"]
    stages = {"N1": (0, 300), "N2": (300, 900)}
    feature_columns = [f"{band}_relative" for band in bands]

    recordings = {path: torch.load(path, map_location="cpu", weights_only=True).detach().cpu().numpy()
                for path in file_paths}

    eeg_indices = list(range(20))
    colours = plt.get_cmap("tab20", 20)

    for stage, interval in stages.items():
        #len(annotations) + 1) // 2, 2, 3.5*((len(file_paths) + 1) // 2)
        fig, axes = plt.subplots(4, 4,figsize=(14, 3.5 *4 ),
                                 squeeze=False, sharex=True, sharey=True )

        for ax, (path, file_name, age_text) in zip(axes.flat, annotations):
            values = recordings[path]

            for channel_index in eeg_indices:

                window_features = extract_band_power(values[channel_index], stage_intervals={stage: interval},
                                                      window_s=win_size, step_s=step_size)

                median_power = (window_features[feature_columns].median().to_numpy())

                ax.plot(bands,median_power,marker="o",linewidth=1,alpha=0.75,
                        color=colours(channel_index),label=montage[channel_index])

            ax.set_title(f"{file_name}, {age_text}")
            ax.tick_params(axis="x", labelbottom=True)
            ax.set_ylabel("Relative band power")
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.25)

        for ax in list(axes.flat)[len(annotations):]:
            ax.set_visible(False)

        handles, legend_labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles,legend_labels, loc="center left", bbox_to_anchor=(0.86, 0.5),fontsize=8)
        fig.suptitle(f"{stage}: median band power across EEG channels", fontweight="bold")
        fig.tight_layout(rect=[0, 0, 0.87, 0.98])

        fig.savefig(save_dir/ f"{stage}_band_power.png", dpi=300, bbox_inches="tight")
        #plt.show()
    return

def plot_absolute_band_pwr(annotations, out_dir, win_size=30, step_size=15):
    '''plot absolute value band power for random participants'''

    file_paths = [path for path, file_name, age_text in annotations]
    save_dir = out_dir / "absolute_band_pwr"/f"win{win_size}_step{step_size}" /f"{len(annotations)/2}per_age_gr"
    os.makedirs(save_dir, exist_ok=True)

    montage = ['FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2','T3-C3', 'C3-CZ',
                'CZ-C4', 'C4-T4','FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1','FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',]
    bands = ["delta", "theta", "alpha", "beta"]
    stages = {"N1": (0, 300), "N2": (300, 900)}

    recordings = {path: torch.load(path, map_location="cpu", weights_only=True).detach().cpu().numpy()
                for path in file_paths}
    
    eeg_indices = list(range(20))
    colours = plt.get_cmap("tab20", 20)

    # power_min = np.inf
    # power_max = -np.inf

    for stage, interval in stages.items():
        all_features = {}

        for path in file_paths:
                all_features[path] = [extract_band_power(recordings[path][channel_index], stage_intervals={stage : interval}, window_s=win_size, step_s=step_size) # type: ignore
                                 for channel_index in eeg_indices]
        for band in bands:          #(len(file_paths) + 1) // 2, 2, 5*(len(file_paths) + 1) // 2)
            
            if band == "delta":
                n = 11
            else:
                n = 5    
            fig, axes = plt.subplots(4, 4, figsize=(15*2, n * 4),
                                     squeeze=False, sharex=True, sharey=True)

            for ax, (path, file_name, age_text) in zip(axes.flat, annotations):

                for channel_index in eeg_indices:
                    channel_features = all_features[path][channel_index]

                    #power = channel_features[f"{band}_absolute"] 
                    #power_min = min(power_min, power.min())
                    #power_max = max(power_max, power.max())

                    s_start, s_end = interval
                    time = channel_features["start_s"] - s_start + win_size/2 # center
                    offset = channel_index*100
                    ax.plot(time, channel_features[f"{band}_absolute"] + offset,linewidth=2,color=colours(channel_index),label=montage[channel_index])
                    
                    ax.set_title(f"{file_name}, {age_text}")                    
                    ax.set_xlabel(f"Seconds from {stage} start")
                    ax.set_ylabel(f"Channels, vertically offset")
                    ax.set_yticks([])
                    if band=="delta":
                        ax.set_ylim(bottom = 0, top=7000)
                    else: 
                        ax.set_ylim(bottom = 0, top=3000)

            handles, legend_labels = axes.flat[0].get_legend_handles_labels()      
            fig.legend(handles,legend_labels, loc="center left", bbox_to_anchor=(0.86, 0.5),fontsize=8)
            #ax.text(0.02, 0.98, f"Power before offset\nMin: {power_min:.3g}\nMax: {power_max:.3g}", transform=ax.transAxes, va="top", fontsize=8,bbox=dict(alpha=0.8, edgecolor="none"),)
            fig.suptitle(f"{stage}: absolute {band} power across EEG channels", fontweight="bold")
            #fig.tight_layout()
            fig.tight_layout(rect=[0, 0, 0.87, 0.98])
            fig.savefig(save_dir/ f"{stage}_absolute_{band}_power.png")      
    return
                     






'''

# Calculating for all participants and saving as a parquet, 
# including absolute and relative band power

structure : recordind_id , channel, stage, start_s, delta_relative

df_spectra
columns = ["rec_id", "channel", "stage", "start", "end", "delta_relative", "theta_relative", "alpha_relative","beta_relative" "delta_abs"]


'''


###########################

ROOT = Path(__file__).resolve().parents[1] #Gervigreind-Project
DATA = ROOT/"data"
OUT = ROOT/"feature_extraction_out"
os.makedirs(OUT, exist_ok=True )

label_df = pd.read_csv(DATA/"anon_codex_age_sex.csv")
eeg_train = torch.load(DATA/"Segments_h1k_train.pt")
eeg_val = torch.load(DATA/"Segments_h1k_val.pt")
eeg_test = torch.load(DATA/"Segments_h1k_test.pt")


# plot_age_dist(df, OUT) 

label_df["age_group"] = pd.cut(label_df['Quantized Age (years)'], bins = [0,2.4,5,10, float("inf")],
                               labels=["0–2.4", ">2.4–5", ">5–10", ">10"],include_lowest=True, right=True)

counts = label_df.groupby("age_group", observed=False).agg(count= ("Anon. Filename", "nunique"))
print(counts)

# PSD plots for few participants. 
previous_names = ["M634P5.edf", "65KT46.edf","Q11N66.edf", "2C4J11.edf", "3NV033.edf", "B217Y7.edf","V022A8.edf","R4404F.edf"]
remaining = label_df[~label_df["Anon. Filename"].isin(previous_names)]
selected = remaining.groupby("age_group", observed=True).sample(n=4, random_state=12)
print(selected)
file_paths = [ DATA/"pre_processed"/f"{Path(name).stem}.pt" for name in selected["Anon. Filename"]]
annotations = []
for path in file_paths:
    match = label_df.loc[label_df["Anon. Filename"] ==  f"{path.stem}.edf", "Quantized Age (years)"]
    age_text = (f"{match.iloc[0]:.2f} years" if len(match) == 1 else "Age match missing or duplicated")
    annotations.append((path, path.stem, age_text))

plot_median_band_pwr(annotations,OUT, win_size=10, step_size=5)
plot_absolute_band_pwr(annotations ,OUT, win_size=10, step_size=5)



'''
notes :
should we split in age groups? Prior work shows that different features and 
different models work best with infants, kids, teens. 
should we split in age groups? 
try first 0–2.4 , >2.4–5, >5–10, >10

split by sleep stages? N1, N2 or  N1+N2?

EEG Frequency bands
 - Delta (0,5–4 Hz) - deep sleep, recovery, unconscious 
 - Theta (4–8 Hz) - light sleep (both N1 and N2), sleepy, meditation, memory encoding
 - Alpha (8–13 Hz) - relaxation, resting, but awake, strong alpha signals =  idle region
 - Beta (13–30 Hz) - active thinking, motor planning, attention/alertness
 - Gamma (30–100+ Hz) - higher cognition, perceptual binding, conscious awareness and information integration

 IN THE END WIN_SIZE =30, STEP=15
 '''
