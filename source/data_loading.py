import numpy as np
import pandas as pd
import os, io
from zipfile import ZipFile
import tarfile
import py7zr
import librosa
import pickle
import gc, joblib, feather
import librosa.display
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from scipy.io import wavfile
import soundfile

data_dir = os.path.join("F:", "Data", "Fourth Year Project")
pickle_dir = "data"
MAPS_pkl_path = os.path.join(pickle_dir, "MAPS", "MAPS.pkl")
BiVib_pkl_path = os.path.join(pickle_dir, "BiVib", "BiVib.pkl")
melspec_pkl_path = os.path.join(pickle_dir, "melspec_preprocessed.pkl")


class TimbreDataset(Dataset):
    # Class to handle dataframe like a torch dataset
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        return row.spectrogram, row.label


class InstrumentLoader:
    def __init__(self, data_dir, note_range=None, set_velocity=None):
        # midi_range, if specified, restricts the notes used in the dataset
        #   we will use C3 to C5 (2 octaves centred around middle C): MIDI 48-72
        # velocity, if specified, restricts the velocities to medium
        self.data_dir = data_dir
        self.dataset = pd.DataFrame()
        self.note_range = note_range
        self.set_velocity = set_velocity

        if not os.path.isfile(MAPS_pkl_path):
            print(MAPS_pkl_path, "not found, loading dataset manually")
            dataset_MAPS = self.load_MAPS(self.note_range, self.set_velocity)
            dataset_MAPS.to_pickle(MAPS_pkl_path)
            print("Pickle saved as", MAPS_pkl_path)
        else:
            print("Loading pickle from", MAPS_pkl_path)
            dataset_MAPS = pd.read_pickle(MAPS_pkl_path)
        self.dataset = self.dataset.append(dataset_MAPS)

        if not os.path.isfile(BiVib_pkl_path):
            print(BiVib_pkl_path, "not found, loading dataset manually")
            dataset_BiVib = self.load_BiVib(self.note_range, self.set_velocity)
            dataset_BiVib.to_pickle(BiVib_pkl_path)
            print("Pickle saved as", BiVib_pkl_path)
        else:
            print("Loading pickle from", BiVib_pkl_path)
            dataset_BiVib = pd.read_pickle(BiVib_pkl_path)
        self.dataset = self.dataset.append(dataset_BiVib)

    def load_MAPS(self, note_range, set_velocity):
        # Types of each piano
        inst_types = {"AkPnBcht": "Grand",
                      "AkPnBsdf": "Grand",
                      "AkPnCGdD": "Grand",
                      "AkPnStgb": "Grand",
                      "SptkBGAm": "Grand",
                      "StbgTGd2": "Grand",
                      "ENSTDkAm": "Upright",
                      "ENSTDkCl": "Upright"}

        zip_path = os.path.join(data_dir, "MAPS", "MAPS.zip")
        data = pd.DataFrame()

        # Extract isolated notes from original MAPS dataset archive
        with ZipFile(zip_path, 'r') as archive:
            for f in archive.namelist():
                if f.endswith(".zip"):
                    # read inner zip file into bytes buffer
                    content = io.BytesIO(archive.read(f))
                    inst_zip = ZipFile(content)
                    for file in inst_zip.namelist():
                        filename = os.path.basename(file)
                        if filename.startswith("MAPS_ISOL_NO") and filename.endswith(".wav"):
                            print("Reading file", file)
                            sample_name = os.path.splitext(filename)[0]
                            # Read labels from file name pattern
                            inst_name = sample_name.split("_")[-1]
                            label = inst_types[inst_name]
                            velocity = sample_name.split("_")[3]
                            sustain = int(sample_name.split("_")[4][-1])

                            # Read annotation info in corresponding .txt
                            with inst_zip.open(os.path.splitext(file)[0] + ".txt") as txtfile:
                                txt = np.genfromtxt(txtfile)
                            start_time = txt[1, 0]
                            end_time = txt[1, 1]
                            pitch = int(txt[1, 2])

                            # Check that pitch falls in the specified range
                            if note_range is not None:
                                if not (note_range[0] <= pitch <= note_range[1]):
                                    continue
                            # Check that velocities match that specified
                            if set_velocity is not None:
                                if velocity != set_velocity:
                                    continue

                            # Assume 44.1kHz sampling rate
                            Fs = 44100
                            # Read waveform
                            wav_file_read = inst_zip.read(file)
                            # audio_data_librosa, Fs = librosa.load(io.BytesIO(wav_file_read), mono=True, sr=None,
                            #                               offset=start_time, duration=end_time-start_time)
                            # Load codec wav PCM s16le 44100Hz using int16 datatype
                            # Fs, audio_data = wavfile.read(io.BytesIO(wav_file_read))
                            audio_data, Fs = soundfile.read(io.BytesIO(wav_file_read), dtype="int16",
                                                      start=int(start_time*Fs), stop=int(end_time*Fs))
                            # Sum to mono without dividing amplitude using 32 bits to prevent 16 bit overflow
                            audio_data = np.sum(audio_data.astype("int32"), axis=1)

                            # Append to dataframe
                            sample_row = pd.DataFrame([["MAPS", inst_name, audio_data, Fs, pitch, velocity, sustain, label]],
                                                      index=pd.Index([sample_name], name="filename"),
                                                      columns=["dataset", "instrument", "waveform", "Fs", "pitch", "velocity", "sustain", "label"])
                            # sample_row = pd.DataFrame({"dataset": pd.Series(["MAPS"], dtype="category"),
                            #                            "instrument": pd.Series([str(inst_name)], dtype="category"),
                            #                            "waveform": [audio_data],
                            #                            "Fs": pd.Series([Fs], dtype="category"),
                            #                            "pitch": pd.Series([pitch], dtype="int8"),
                            #                            "velocity": pd.Series([velocity], dtype="category"),
                            #                            "sustain": pd.Series([sustain], dtype=bool),
                            #                            "label": pd.Series([label], dtype="category")})
                            # sample_row.index = pd.Index([sample_name], name="filename")

                            data = data.append(sample_row)
                    print("Loaded", inst_name)
        return data

    def load_NSynth(self):
        test_tar_path = os.path.join(data_dir, "NSynth", "nsynth-test.jsonwav.tar.gz")
        train_tar_path = os.path.join(data_dir, "NSynth", "nsynth-train.jsonwav.tar.gz")
        valid_tar_path = os.path.join(data_dir, "NSynth", "nsynth-valid.jsonwav.tar.gz")
        load_path = os.path.join(data_dir, "NSynth", "extracted")

        for subset_tar_path in [test_tar_path, train_tar_path, valid_tar_path]:
            print("Extracting", subset_tar_path)
            with tarfile.open(subset_tar_path, 'r') as archive:
                for member in archive.getmembers():
                    filename = os.path.basename(member.name)
                    if filename == "examples.json"\
                            or filename.startswith("keyboard_acoustic")\
                            or filename.startswith("keyboard_electronic"):
                        # We manually exclude mislabelled/non-piano sounds
                        # since we want only acoustic and electroacoustic pianos
                        if not (filename.startswith("keyboard_acoustic_011")
                             or filename.startswith("keyboard_acoustic_015")
                             or filename.startswith("keyboard_acoustic_017")
                             or filename.startswith("keyboard_acoustic_020")
                             or filename.startswith("keyboard_electronic_005")
                             or filename.startswith("keyboard_electronic_006")
                             or filename.startswith("keyboard_electronic_012")
                             or filename.startswith("keyboard_electronic_014")
                             or filename.startswith("keyboard_electronic_021")
                             or filename.startswith("keyboard_electronic_022")
                             or filename.startswith("keyboard_electronic_023")
                             or filename.startswith("keyboard_electronic_024")
                             or filename.startswith("keyboard_electronic_025")
                        ):
                            archive.extract(member, load_path)
        # keyboard_acoustic_003 has to be relabelled as electric

    def extract_BiVib(self):
        zip_path = os.path.join(self.data_dir, "BiVib", "downloaded")
        temp_zip = os.path.join(self.data_dir, "BiVib", "tmp_grand.7z")

        zip_names = [f for f in os.listdir(zip_path) if f.startswith("Grand.7z")]
        if not os.path.isfile(temp_zip):
            # Join multi-part archive
            with open(temp_zip, 'ab') as tmparch:  # append in binary mode
                for fname in zip_names:
                    with open(os.path.join(zip_path, fname), 'rb') as infile:
                        tmparch.write(infile.read())
        # Extract binaural sample wavs from joined archive
        # with py7zr.SevenZipFile(temp_zip, "r") as tmparch:
        #     binaural_samples = [f for f in tmparch.getnames()
        #                         if f.startswith("Upright/binaural samples") and f.endswith(".wav")]
        #     files_dict = tmparch.read(binaural_samples)
            # for file in binaural_samples:
            #     print("Reading", file)
            #     tmparch.reset()
            #     file_dict = tmparch.read([file])
            #     filename = next(iter(file_dict))
            # for filename, wavfile in files_dict.items():
            #     sample_name = os.path.splitext(filename)[0]
            #     basename = os.path.basename(filename)
            #     note_name = basename.split("_")[0]
            #     pitch = librosa.note_to_midi(note_name)
            #     velocity = basename.split("_")[1]
            #     sustain = 0
            #     label = "Upright"
            #
            #     # wavfile = next(iter(file_dict.values()))
            #     audio_data, Fs = librosa.load(wavfile, mono=True)
            #     # Append to dataframe
            #     sample_row = pd.DataFrame([[audio_data, pitch, velocity, sustain, label]],
            #                               index=pd.Index([sample_name], name="filename"),
            #                               columns=["waveform", "pitch", "velocity", "sustain", "label"])
            #     data = data.append(sample_row)
            #     # Reset read pointer
            #     wavfile.seek(0)

    def load_BiVib(self, note_range, set_velocity):
        data = pd.DataFrame()
        extracted_path = os.path.join(self.data_dir, "BiVib", "extracted")
        sustain = 0 # all samples are played without sustain pedal
        for label in ["Upright", "Grand"]:
            label_subpath = os.path.join(extracted_path, label)
            for instrument_type in os.listdir(label_subpath):
                for filename in os.listdir(os.path.join(label_subpath, instrument_type)):
                    file_path = os.path.join(label_subpath, instrument_type, filename)
                    print("Reading file", file_path)
                    sample_name = os.path.splitext(filename)[0]
                    note_name = sample_name.split("_")[0]
                    pitch = librosa.note_to_midi(note_name)
                    velocity_midi = int(sample_name.split("_")[1])
                    if velocity_midi < 56:
                        velocity = "P"
                    elif 56 <= velocity_midi <= 78:
                        velocity = "M"
                    else:
                        velocity = "F"

                    # Check that pitch falls in the specified range
                    if note_range is not None:
                        if not (note_range[0] <= pitch <= note_range[1]):
                            continue
                    # Check that velocities match those specified
                    if set_velocity is not None:
                        if velocity != set_velocity:
                            continue

                    # BiVib is 96kHz 24 bit int but we want to convert to CD quality 44100 Hz int16
                    #audio_data, Fs = librosa.load(file_path, mono=True, sr=None, dtype="int16")
                    audio_data, orig_Fs = soundfile.read(file_path, dtype="int16")
                    # Convert from stereo to mono by summing channels.
                    # The division and sum are performed as float64 to prevent overflow and truncation
                    audio_data = np.sum(audio_data/2, axis=1)
                    # Resample to 44.1kHz. scipy.signal.resample may be faster, uses Fourier domain
                    Fs = 44100
                    audio_data = librosa.resample(audio_data.astype(float), orig_Fs, Fs).astype("int32")
                    # Crop to 2.1s to match MAPS lengths, since BiVib holds note until it dies out while MAPS releases after about 2.1s
                    if len(audio_data) > 97285:
                        audio_data = audio_data[:97285]

                    # Append to dataframe
                    sample_row = pd.DataFrame({"dataset": pd.Series(["BiVib"], dtype="category"),
                                               "instrument": pd.Series([str(instrument_type)], dtype="category"),
                                               "waveform": [audio_data],
                                               "Fs": pd.Series([Fs], dtype="category"),
                                               "pitch": pd.Series([pitch], dtype="int8"),
                                               "velocity": pd.Series([velocity], dtype="category"),
                                               "sustain": pd.Series([sustain], dtype=bool),
                                               "label": pd.Series([label], dtype="category")})
                    sample_row.index = pd.Index([sample_name], name="filename")
                    data = data.append(sample_row)
        return data

    def stack_velocities(self, data):
        out = pd.DataFrame()
        # Get unique info for each note sample, ignoring velocity and sustain information
        #unique_notes = np.unique(np.vstack(np.char.split(np.array(data.index).astype(str), "_S", maxsplit=1))[:, -1])
        unique_notes = np.unique(np.column_stack((data["instrument"].to_numpy(str),
                                                  data["pitch"].to_numpy(str))), axis=0)
        for note_sample in unique_notes:
            # Stack the spectrograms of the 3 velocity layers for each note of each instrument
            velocity_layers = data[(data.instrument == note_sample[0]) & (data.pitch == note_sample[1].astype(int))]
            # Ensure stacked layers are in alphabetical order F-M-P (descending vel.) for consistency
            velocity_layers = velocity_layers.sort_values("velocity")
            stacked_layers = np.stack((velocity_layers.spectrogram))
            # append dataframe with 3d spectrograms for each note
            dataset = velocity_layers.dataset.iloc[0]
            instrument_name = note_sample[0]
            framerate = velocity_layers.framerate.iloc[0]
            pitch = note_sample[1].astype(int)
            label = velocity_layers.label.iloc[0]
            out = out.append(pd.DataFrame({"dataset": dataset,
                                           "instrument": instrument_name,
                                           "spectrogram": [stacked_layers],
                                           "framerate": framerate,
                                           "pitch": pitch,
                                           "label": label}, index=[0]))
        return out

    def preprocess(self,                # STFT and mel-spectrogram parameters assuming Fs=44100:
                   n_fft=2048,              # 46 ms STFT frame length
                   win_length=0.025,        # 25 ms spectrogram frame length, 0-padded to apply STFT over 46 ms
                   window_spacing=0.010,    # 10 ms hop size between windows
                   window="hamming",
                   n_mels=300,              # No. of mel filter bank bands (y-axis resolution) - hyper-parameter
                   fmin=0, fmax=None,       # 20-8000 Hz is piano's perceptible range
                                            #  results in 300 bins, each 66.6 Hz wide (NO NOT LINEAR SINCE MEL SCALE - Plot the mel filter)
                   vel_stack=False, crop=False, pad=True,
                   normalisation=None,
                   plot=False):
        if not os.path.isfile(melspec_pkl_path):
            print(melspec_pkl_path, "not found, pre-processing dataset manually")

            max_len = len(max(self.dataset["waveform"], key=len))
            spec_params = {"Fs": 44100, # NOTE: this is the waveform's sample rate, not the spectrogram framerate.
                                        # We assume this is always 44100Hz for the purposes of plotting
                           "framerate": 1 / window_spacing,
                           "fmin": fmin,
                           "fmax": fmax}
            # Plot mel filter bank used to generate the mel spectrogram
            plt.figure(figsize=(10, 12))
            plt.subplot(2, 1, 1)
            filterbank = librosa.filters.mel(spec_params["Fs"], n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, norm=None)
            img = librosa.display.specshow(filterbank, x_axis='linear', y_axis="mel", sr=spec_params["Fs"], fmin=fmin, fmax=fmax)
            plt.xlabel("STFT Frequencies (Hz)")
            plt.ylabel('Mel filter frequencies (Hz)')
            plt.title('Mel filter bank')
            plt.colorbar(img, label="Filter Magnitude")
            # Plotting each triangular filter's reponse:
            #  reference: https://stackoverflow.com/questions/40197060/librosa-mel-filter-bank-decreasing-triangles
            plt.subplot(2, 1, 2)
            selected_bank_indices = np.linspace(start=0, stop=n_mels-1, num=75, dtype=int)
            for i in selected_bank_indices:
                plt.plot(np.linspace(fmin, fmax, int((n_fft/2)+1)), filterbank[i])
            #plt.legend(labels=selected_bank_indices)
            plt.title("Frequency response of selected normalised Mel bank filters")
            plt.ylim([0.00001, None])
            plt.xlim([0, 18500])
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Magnitude")
            plt.show()

            out = pd.DataFrame()
            for i, sample in self.dataset.iterrows():
                waveform = sample["waveform"]
                Fs = sample["Fs"]

                # Convert from int32 to float32 between -1 and 1 for librosa processing
                if waveform.dtype == "int32":
                    waveform = np.array([np.float32((s >> 1) / (32768.0)) for s in waveform])
                if pad:
                    # 0-pad waveforms to the maximum waveform length in the dataset
                    waveform = np.pad(waveform, (0, max_len-len(waveform)))
                if fmax is None:
                    # Use Nyquist frequency if no max frequency is specified
                    fmax = Fs/2
                melspec = librosa.feature.melspectrogram(waveform, Fs,
                                                         n_fft=n_fft,                   # Samples per STFT frame
                                                         win_length=int(win_length*Fs), # Samples per 0-padded spec window
                                                         hop_length=int(window_spacing*Fs), # No. of samples between windows
                                                         window=window,                 # Window type
                                                         n_mels=n_mels,                 # No. of mel freq bins
                                                         fmin=fmin, fmax=fmax)

                # Convert to log power scale
                melspec = librosa.power_to_db(melspec, ref=np.max)
                # Temporary solution: remove last few samples to make all spectrograms square and the same length
                if crop:
                    melspec = melspec[:, :172]

                if plot:
                    plot_spectrogram(melspec, spec_params, name=str(sample.name))

                if normalisation == "statistics":
                    melspec = (melspec - np.mean(melspec))/np.std(melspec)

                # Magnitude-normalise spectrogram by dividing by the fundamental frequency's magnitude
                elif normalisation == "fundamental":
                    mel_freq_axis = librosa.mel_frequencies(n_mels=n_mels,fmin=fmin,fmax=fmax)

                    # Use the magnitude of the bin containing the fundamental frequency (with center nearest to fundamental)
                    fundamental_freq = librosa.core.midi_to_hz(sample["pitch"])
                    fundamental_bin_index = (np.abs(mel_freq_axis - fundamental_freq)).argmin()
                    # Divide by peak magnitude
                    fundamental_bin_mag = np.max(melspec[fundamental_bin_index, :])
                    print("Nearest bin to the fundamental is at", mel_freq_axis[fundamental_bin_index],
                          "Hz, with peak magnitude", fundamental_bin_mag)
                    melspec = melspec/fundamental_bin_mag

                # Convert labels to binary, "Grand" = 0, "Upright" = 1
                if sample["label"] == "Grand":
                    label = 0
                if sample["label"] == "Upright":
                    label = 1

                spec_row = pd.DataFrame({"dataset": sample["dataset"],
                                         "instrument": sample["instrument"],
                                         "spectrogram": [melspec],
                                         "pitch": sample["pitch"],
                                         "velocity": sample["velocity"],
                                         "sustain": sample["sustain"],
                                         "label": label})
                spec_row.index = pd.Index([sample.name], name="filename")
                out = out.append(spec_row)
            if vel_stack:
                out = self.stack_velocities(out)
            out.to_pickle(melspec_pkl_path)
            print("Pre-processed mel-spectrograms saved to", melspec_pkl_path)
        else:
            print("Loading pickle from", melspec_pkl_path)
            out = pd.read_pickle(melspec_pkl_path)

        return out


def plot_spectrogram(spectrogram, spec_params, name=""):
    fig, ax = plt.subplots()
    img = librosa.display.specshow(spectrogram, x_axis='s', y_axis='mel', ax=ax,
                                   sr=spec_params["Fs"],
                                   hop_length=int(spec_params["Fs"]/spec_params["framerate"]),
                                   fmin=spec_params["fmin"],
                                   fmax=spec_params["fmax"])
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram: '+name)
    plt.show()


if __name__ == '__main__':
    loader = InstrumentLoader(data_dir, note_range=[48, 72], set_velocity="M")

    upright_count = len(loader.dataset.loc[loader.dataset['label'] == "Upright"])
    grand_count = len(loader.dataset.loc[loader.dataset['label'] == "Grand"])

    # Shape: (1, 300, 221)
    melspec_data = loader.preprocess(fmin=20, fmax=20000, n_mels=300, normalisation="statistics", plot=False)
    print("")
