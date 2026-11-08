import os
from pathlib import Path
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

#os.getcwd
ROOT = Path(__file__).resolve().parents[1] #Gervigreind-Prjoect
DATA = ROOT/"data"
OUT = ROOT/"feature_selection"
os.makedirs(OUT, exist_ok=True )

label_df = pd.read_csv(DATA/"anon_codex_age_sex.csv")
eeg_train = torch.load(DATA/"Segments_h1k_train.pt")
eeg_val = torch.load(DATA/"Segments_h1k_val.pt")
eeg_test = torch.load(DATA/"Segments_h1k_test.pt")

# should we split in age groups? Prior work shows that different features and 
# different models work best with infants, kids, teens. 

plt.figure(figsize=(9,4))
plt.hist(label_df["Quantized Age (years)"], bins = np.arange(0,18,1))
plt.xlabel("Age")
plt.ylabel("Count")
plt.title("Participant age distribution")
#plt.savefig(OUT/"age_dist.pdf")
#plt.show()



# try first 0–2.4 , >2.4–5, >5–10, >10
label_df["age_group"] = pd.cut(label_df['Quantized Age (years)'], bins = [0,2.4,5,10, float("inf")],
                               labels=["0–2.4", ">2.4–5", ">5–10", ">10"],include_lowest=True, right=True)


counts = label_df.groupby("age_group", observed=False).agg(count= ("Anon. Filename", "nunique"))
print(counts)

#age = label_df["Quantized Age (years)"]
# for upper_age in [2.4, 3, 4]:           
#     n = label_df.loc[ age.between(0, upper_age), "Anon. Filename"].nunique()
#     print(f"0–{upper_age} years: {n} participants")



#Stage specific? N1 + N2 / N1 / N2 ?


# Band power: Calculate power spectrum and then intergratet over each band.
# EEG Frequency bands
# Delta (0,5–4 Hz) - deep sleep, recovery, unconsious 
# Theta (4–8 Hz) - light sleep (both N1 and N2), sleepy, meditation, memory encoding
# Alpha (8–13 Hz) - relaxation, resting, but awake, strong alpha signals =  idle region
# Beta (13–30 Hz) - active thinking, motor planning, attention/alertness
#  Gamma (30–100+ Hz) - higher cognition, perceptual binding, conscious awareness and information integration

def extract_band_power(signal, stage_intervals, sfreq = 64, window_s=60, step_s=50):

    bands = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 12), "beta": (12, 30)}

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
                return np.trapezoid(p, f)

            total_pwr = power_between(0.5,30)

            row = {"stage": stage, "start_s": start / sfreq}
            for band, (low, high) in bands.items():
                absolute = power_between(low, high)
                row[f"{band}_absolute"] = absolute
                row[f"{band}_relative"] = absolute / total_pwr

            rows.append(row)
            
    return pd.DataFrame(rows)

bands = ["delta", "theta", "alpha", "beta"]
feature_columns = [f"{band}_relative" for band in bands]

stages = {"N1": (0, 300), "N2": (300, 900)}


selected = label_df.groupby("age_group", observed=True).sample(n=2, random_state=12)
print(selected)

file_paths = [ DATA/"pre_processed"/f"{Path(name).stem}.pt" for name in selected["Anon. Filename"]]
recordings = {path: torch.load(path, map_location="cpu", weights_only=True).detach().cpu().numpy()
              for path in file_paths}


eeg_indices = list(range(20))
colours = plt.get_cmap("tab20", 20)

for stage, interval in stages.items():

    fig, axes = plt.subplots(
        (len(file_paths) + 1) // 2, 2,
        figsize=(14, 3.5 * ((len(file_paths) + 1) // 2)),
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    for ax, path in zip(axes.flat, file_paths):
        values = recordings[path]

        for channel_index in eeg_indices:

            window_features = extract_band_power(
                values[channel_index],
                sfreq=64,
                stage_intervals={stage: interval},
                window_s=60,
                step_s=30,
            )

            median_power = (
                window_features[feature_columns]
                .median()
                .to_numpy()
            )

            ax.plot(
                bands,
                median_power,
                marker="o",
                linewidth=1,
                alpha=0.75,
                color=colours(channel_index),
                label=f"Channel {channel_index + 1}",
            )

        match = label_df.loc[label_df["Anon. Filename"] == f"{path.stem}.edf", "Quantized Age (years)",]

        age_text = (f"{match.iloc[0]:.2f} years" if len(match) == 1 else "Age match missing or duplicated")

        ax.set_title(f"{path.stem}\n{age_text}")
        ax.tick_params(axis="x", labelbottom=True)
        ax.set_ylabel("Relative band power")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)

    for ax in list(axes.flat)[len(file_paths):]:
        ax.set_visible(False)

    handles, legend_labels = axes.flat[0].get_legend_handles_labels()

    fig.legend(handles,legend_labels, loc="center left", bbox_to_anchor=(0.86, 0.5),fontsize=8,)
    fig.suptitle(f"{stage}: median band power across EEG channels")
    fig.tight_layout(rect=[0, 0, 0.85, 0.96])
    fig.savefig(
    OUT / f"{stage}_band_power.png",
    dpi=300,
    bbox_inches="tight")
    #plt.show()