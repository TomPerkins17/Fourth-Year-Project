import numpy as np
import pandas as pd
import os, io
from zipfile import ZipFile
import tarfile
import librosa
import pickle
import librosa.display
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import soundfile
import sounddevice as sd
import mido
import time

data_dir = os.path.join("F:", "Data", "Fourth Year Project")
pickled_data_dir = "data"
sample_new_midi_inst = False


class InstrumentLoader:
    def __init__(self, data_dir, note_range=None, set_velocity=None, normalise_wavs=True, load_MIDIsampled=True,
                 reload_wavs=False):
        # midi_range, if specified, restricts the notes used in the dataset
        #   C3 to C5 (2 octaves centred around middle C): MIDI 48-72
        # velocity, if specified, restricts the velocities to medium
        self.data_dir = data_dir
        self.dataset = pd.DataFrame()
        self.note_range = note_range
        self.set_velocity = set_velocity
        self.Fs = 44100

        # Set up pickle filepaths for different versions of the dataset
        normalisedwavs_field = "normalisedwavs" if normalise_wavs else "no-norm"
        velocity_field = "all" if set_velocity is None else set_velocity
        MIDIsampled_field = "+MIDIsampled" if load_MIDIsampled else ""
        MAPS_pkl = os.path.join(pickled_data_dir, "single-note_dataset", "MAPS",
                                "MAPS" + "_" + normalisedwavs_field + "_" + velocity_field + ".pkl")
        BiVib_pkl = os.path.join(pickled_data_dir, "single-note_dataset", "BiVib",
                                 "BiVib" + "_" + normalisedwavs_field + "_" + velocity_field + ".pkl")
        self.melspec_pkl = os.path.join(pickled_data_dir, "single-note_dataset", "preprocessed",
                                        "melspec_preprocessed"+"_"+normalisedwavs_field+"_"+velocity_field+MIDIsampled_field+".pkl")

        # Speed up loading and reduce mem usage when pre-processed pickles are already available and wavs not needed
        if reload_wavs:
            if not os.path.isfile(MAPS_pkl):
                print(MAPS_pkl, "not found, loading dataset manually")
                dataset_MAPS = self.load_MAPS(self.note_range, self.set_velocity, normalise=normalise_wavs)
                dataset_MAPS.to_pickle(MAPS_pkl)
                print("Pickle saved as", MAPS_pkl)
            else:
                print("Loading pickle from", MAPS_pkl)
                dataset_MAPS = pd.read_pickle(MAPS_pkl)
            self.dataset = self.dataset.append(dataset_MAPS)

            if not os.path.isfile(BiVib_pkl):
                print(BiVib_pkl, "not found, loading dataset manually")
                dataset_BiVib = self.load_BiVib(self.note_range, self.set_velocity, normalise=normalise_wavs)
                dataset_BiVib.to_pickle(BiVib_pkl)
                print("Pickle saved as", BiVib_pkl)
            else:
                print("Loading pickle from", BiVib_pkl)
                dataset_BiVib = pd.read_pickle(BiVib_pkl)
            self.dataset = self.dataset.append(dataset_BiVib)

            if load_MIDIsampled:
                dataset_MIDIsampled = self.load_MIDIsampled_dataset(pkl_dir="data/single-note_dataset/MIDIsampled", note_range=note_range, sample_new=sample_new_midi_inst)
                self.dataset = self.dataset.append(dataset_MIDIsampled)

    def load_MAPS(self, note_range, set_velocity, normalise):
        # Types of each piano
        inst_types = {"AkPnBcht": "Grand",
                      "AkPnBsdf": "Grand",
                      "AkPnCGdD": "Grand",
                      "AkPnStgb": "Upright",
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
                            # Check that velocities match those specified
                            if set_velocity is not None:
                                if velocity != set_velocity:
                                    continue

                            # Read waveform
                            wav_file_read = inst_zip.read(file)
                            # Load codec wav PCM s16le 44100Hz using int16 datatype MAPS has a 44.1kHz sampling rate
                            audio_data, read_Fs = soundfile.read(io.BytesIO(wav_file_read), dtype="int16",
                                                      start=int(start_time*self.Fs), stop=int(end_time*self.Fs))
                            if read_Fs != self.Fs:
                                raise Exception("Mismatch between loader sample rate of " + str(self.Fs) +
                                                " and file sample rate of " + str(read_Fs))
                            # Retain only the L channel to get mono wav information
                            audio_data = audio_data[:, 0].astype(np.int32)
                            # Sum to mono without dividing amplitude using 32 bits to prevent 16 bit overflow - Not used due to phase cancellation
                            # audio_data = np.sum(audio_data.astype("int32"), axis=1)

                            # Normalise amplitude to make volume uniform across different notes
                            if normalise:
                                audio_data = audio_data - np.mean(audio_data).astype(np.int32)   # Remove DC offset
                                audio_data = (2147483647*(audio_data/np.max(np.abs(audio_data)))).astype(np.int32)

                            # Append to dataframe
                            sample_row = pd.DataFrame([["MAPS", inst_name, audio_data, self.Fs, pitch, velocity, sustain, label]],
                                                      index=pd.Index([sample_name], name="filename"),
                                                      columns=["dataset", "instrument", "waveform", "Fs", "pitch", "velocity", "sustain", "label"])
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

    def load_BiVib(self, note_range, set_velocity, normalise):
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

                    # BiVib wavs are 96kHz 24 bit int, we want to convert to CD quality 44100 Hz int16
                    audio_data, orig_Fs = soundfile.read(file_path, dtype="int32")
                    # Retain only the L channel to get mono wav information
                    audio_data = audio_data[:, 0]

                    # Convert from stereo to mono by summing channels - Not optimal due to phase cancellation
                    #audio_data = np.sum(audio_data/2, axis=1) # Division and sum are performed as float64 to prevent overflow and truncation
                    # Resample to loader sample rate. scipy.signal.resample may be faster, uses Fourier domain
                    audio_data = librosa.resample(audio_data.astype(float), orig_Fs, self.Fs).astype("int32")

                    # Normalise amplitude to make volume uniform across different notes
                    if normalise:
                        audio_data = audio_data - np.mean(audio_data)  # Remove DC offset
                        audio_data = (2147483647 * (audio_data / np.max(np.abs(audio_data)))).astype(np.int32)

                    # Crop to 2.1s to match MAPS lengths, since BiVib holds note until it dies out while MAPS releases after about 2.1s
                    if len(audio_data) > 97285:
                        audio_data = audio_data[:97285]

                    # Append to dataframe
                    sample_row = pd.DataFrame({"dataset": pd.Series(["BiVib"], dtype="category"),
                                               "instrument": pd.Series([str(instrument_type)], dtype="category"),
                                               "waveform": [audio_data],
                                               "Fs": pd.Series([self.Fs], dtype="category"),
                                               "pitch": pd.Series([pitch], dtype="int8"),
                                               "velocity": pd.Series([velocity], dtype="category"),
                                               "sustain": pd.Series([sustain], dtype=bool),
                                               "label": pd.Series([label], dtype="category")})
                    sample_row.index = pd.Index([sample_name], name="filename")
                    data = data.append(sample_row)
        return data

    def load_MIDIsampled_dataset(self, pkl_dir, note_range=None, sample_new=False):
        data = pd.DataFrame()
        if not sample_new:
            for pkl_name in os.listdir(pkl_dir):
                instrument_pkl_path = os.path.join(pkl_dir, pkl_name)
                print("Loading pickle from", instrument_pkl_path)
                instrument_samples = pd.read_pickle(instrument_pkl_path)
                data = data.append(instrument_samples)
        else:
            print("Manually sampling dataset from instruments")
            continue_sampling = input("Enter \"y\" if you want to, and are ready to, sample an instrument") == "y"
            while continue_sampling:
                instrument_name = input("Enter instrument name")
                instrument_label = input("Enter instrument label")
                instrument_samples = self.load_midi_instrument(instrument_name, instrument_label, note_range, pkl_dir, plot=False)
                data = data.append(instrument_samples)
                continue_sampling = input("Enter \"y\" if you want to, and are ready to, sample an instrument") == "y"
        return data

    def load_midi_instrument(self, instrument_name=None, instrument_label=None, note_range=None, pkl_dir="data/single-note_dataset/MIDIsampled", plot=False):
        rec_offset = 1  # integer number of seconds to wait after starting to record before sending MIDI message
        duration = 2.21 + rec_offset  # recording duration in seconds
        velocity_range = np.linspace(start=0, stop=127, num=5).astype(int)[
                         1:4]  # Creates 3 evenly spaced values around 64
        # Should detect audio interface midi output named "Focusrite USB MIDI 1"
        print("Available midi ports:", mido.get_output_names())
        outport = mido.open_output("Focusrite USB MIDI 1")

        data = pd.DataFrame()
        for note_pitch in range(note_range[0], note_range[1] + 1):
            for velocity_midi in velocity_range:
                # Make sure sound device (audio interface) has sample rate set to self.Fs = 44100 Hz
                note_wav = sd.rec(int(duration * self.Fs), samplerate=self.Fs, channels=1)
                time.sleep(
                    rec_offset)  # Ensure release from previous note doesn't bleed into current one + avoid artifacts
                outport.send(mido.Message("note_on", note=note_pitch, velocity=velocity_midi))
                sd.wait()
                outport.send(mido.Message("note_off", note=note_pitch))
                note_wav = note_wav.flatten()
                note_wav = note_wav[int(rec_offset * self.Fs):]     # Compensate for recording offset
                note_wav = note_wav - np.mean(note_wav)             # Remove waveform DC offset
                note_wav = note_wav / np.max(np.abs(note_wav))      # Normalise waveform
                if plot:
                    plt.plot(note_wav)
                    plt.show()

                # Convert to string velocities
                if velocity_midi == velocity_range[0]:
                    velocity = "P"
                if velocity_midi == velocity_range[1]:
                    velocity = "M"
                if velocity_midi == velocity_range[2]:
                    velocity = "F"

                # soundfile.write("data/Roland_FP80_samples/"+str(note_pitch)+"_"+str(velocity)+".wav", note_wav, self.Fs)

                sample_row = pd.DataFrame({"dataset": pd.Series(["MIDIsampled"], dtype="category"),
                                           "instrument": pd.Series([instrument_name], dtype="category"),
                                           "waveform": [note_wav],
                                           "Fs": pd.Series([self.Fs], dtype="category"),
                                           "pitch": pd.Series([note_pitch], dtype="int8"),
                                           "velocity": pd.Series([velocity], dtype="category"),
                                           "sustain": pd.Series([0], dtype=bool),
                                           "label": pd.Series([instrument_label], dtype="category")})
                data = data.append(sample_row)
        pickle_path = os.path.join(pkl_dir, instrument_name + ".pkl")
        data.to_pickle(pickle_path)
        print("Pickle saved as", pickle_path)
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

    def preprocess(self,                # STFT and mel-spectrogram parameters assuming self.Fs=44100:
                   n_fft=2048,              # 46 ms STFT frame length
                   win_length=0.025,        # 25 ms spectrogram frame length, 0-padded to apply STFT over 46 ms
                   window_spacing=0.010,    # 10 ms hop size between windows
                   window="hamming",
                   n_mels=300,              # No. of mel filter bank bands (y-axis resolution) - hyper-parameter
                   fmin=20, fmax=20000,     # 20-8000 Hz is piano's perceptible range
                   vel_stack=False, pad=True,
                   normalisation="statistics",
                   plot=False, reload_melspec=False):
        if (not os.path.isfile(self.melspec_pkl)) or reload_melspec:
            print(self.melspec_pkl, "not found, pre-processing dataset manually")

            max_len = len(max(self.dataset["waveform"], key=len))
            spec_params = {"Fs": self.Fs,   # NOTE: this is the waveform's sample rate, not the spectrogram framerate.
                           "framerate": 1 / window_spacing,
                           "fmin": fmin,
                           "fmax": fmax,
                           "n_fft": n_fft,
                           "n_mels": n_mels}
            # Plot mel filter bank used to generate the mel spectrogram
            plot_filterbank(spec_params)

            out = pd.DataFrame()
            for i, sample in self.dataset.iterrows():
                waveform = sample["waveform"]
                sample_Fs = sample["Fs"]

                if fmax is None:
                    # Use Nyquist frequency if no max frequency is specified
                    fmax = sample_Fs/2

                if pad:
                    # 0-pad waveforms to the maximum waveform length in the dataset
                    waveform = np.pad(waveform, (0, max_len-len(waveform)))

                # Convert labels to binary, "Grand" = 0, "Upright" = 1
                if sample["label"] == "Grand":
                    label = 0
                if sample["label"] == "Upright":
                    label = 1

                # Compute the log-mel spectrogram
                melspec = self.compute_spectrogram(waveform, sample_Fs, n_fft, win_length, window_spacing, window, n_mels, fmin, fmax, plot)

                # Normalise the spectrogram's magnitudes
                if normalisation == "statistics":
                    melspec = (melspec - np.mean(melspec)) / np.std(melspec)

                elif normalisation == "fundamental":
                    # Magnitude-normalise spectrogram by dividing by the fundamental frequency's magnitude
                    mel_freq_axis = librosa.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax)

                    # Use the magnitude of the bin containing the fundamental frequency (center nearest to fundamental)
                    fundamental_freq = librosa.core.midi_to_hz(sample["pitch"])
                    fundamental_bin_index = (np.abs(mel_freq_axis - fundamental_freq)).argmin()
                    # Divide by peak magnitude
                    fundamental_bin_mag = np.max(melspec[fundamental_bin_index, :])
                    print("Nearest bin to the fundamental is at", mel_freq_axis[fundamental_bin_index],
                          "Hz, with peak magnitude", fundamental_bin_mag)
                    melspec = melspec / fundamental_bin_mag

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
            out.to_pickle(self.melspec_pkl)
            print("Pre-processed mel-spectrograms saved to", self.melspec_pkl)
        else:
            print("Loading pickle from", self.melspec_pkl)
            out = pd.read_pickle(self.melspec_pkl)

        return out

    def compute_spectrogram(self, waveform, waveform_Fs,
                            n_fft=2048,              # 46 ms STFT frame length
                            win_length=0.025,        # 25 ms spectrogram frame length, 0-padded to apply STFT over 46 ms
                            window_spacing=0.010,    # 10 ms hop size between windows
                            window="hamming",
                            n_mels=300,              # No. of mel filter bank bands (y-axis resolution) - hyper-parameter
                            fmin=20, fmax=20000,
                            plot=False):
        # Convert from int32 to float32 between -1 and 1 for librosa processing
        if waveform.dtype == "int32":
            waveform = np.array([np.float32((s >> 1) / (32768.0)) for s in waveform])

        melspec = librosa.feature.melspectrogram(waveform, waveform_Fs,
                                                 n_fft=n_fft,  # Samples per STFT frame
                                                 win_length=int(win_length * waveform_Fs),  # Samples per 0-padded spec window
                                                 hop_length=int(window_spacing * waveform_Fs),  # No. of samples between windows
                                                 window=window,  # Window type
                                                 n_mels=n_mels,  # No. of mel freq bins
                                                 fmin=fmin, fmax=fmax)
        # Convert to log power scale
        melspec = librosa.power_to_db(melspec, ref=np.max)

        if plot:
            plot_spectrogram(melspec, {"Fs": waveform_Fs, "framerate": 1/window_spacing, "fmin": fmin, "fmax": fmax})

        return melspec


def plot_filterbank(spec_params):
    Fs = spec_params["Fs"]
    n_fft = spec_params["n_fft"]
    n_mels = spec_params["n_mels"]
    fmin = spec_params["fmin"]
    fmax = spec_params["fmax"]

    filterbank = librosa.filters.mel(Fs, n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, norm=None)
    # Plotting the frequency mapping of the filterbank
    fig1 = plt.figure(figsize=(10, 5))
    img = librosa.display.specshow(filterbank, x_axis='linear', y_axis="mel", sr=Fs, fmin=fmin,
                                   fmax=fmax)
    plt.xlabel("STFT Frequencies (Hz)")
    plt.ylabel('Mel filter frequencies (Hz)')
    plt.title('Mel filter bank: frequency mapping and magnitude')
    plt.colorbar(img, label="Filter Magnitude")
    plt.show()

    # Plotting each triangular filter's reponse:
    #  reference: https://stackoverflow.com/questions/40197060/librosa-mel-filter-bank-decreasing-triangles
    fig2 = plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.suptitle("Details of the frequency response of a " + str(n_mels) + "-filter Mel filterbank", size='x-large')
    plt.subplots_adjust(hspace=0.5)
    plt.plot(np.linspace(fmin, fmax, int((n_fft / 2) + 1)), filterbank.T)
    plt.title("Detail: 5000 Hz to 7000 Hz")
    plt.ylim([0.00001, None])
    plt.xlim([5000, 7000])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.subplot(2, 1, 2)
    plt.plot(np.linspace(fmin, fmax, int((n_fft / 2) + 1)), filterbank.T)
    plt.title("Detail: 15000 Hz to 17000 Hz")
    plt.ylim([0.00001, None])
    plt.xlim([15000, 17000])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.show()
    # fig1.savefig("../Figures/Mel_mapping.svg")
    # fig2.savefig("../Figures/Mel_filterbank.svg")


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
    #fig.savefig("../Figures/Sample_spectrogram.svg")


class TimbreDataset(Dataset):
    # Class to handle dataframe like a torch dataset
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        return row.spectrogram, row.label, row.instrument


if __name__ == '__main__':
    loader = InstrumentLoader(data_dir, note_range=[48, 72], set_velocity=None, normalise_wavs=True, load_MIDIsampled=True, reload_wavs=True)
    melspec_data = loader.preprocess(fmin=20, fmax=20000, n_mels=300, normalisation="statistics", plot=False, reload_melspec=False)

    sample_melspec = melspec_data.iloc[0].spectrogram
    plot_spectrogram(sample_melspec, {"Fs": 44100,   # NOTE: this is the waveform's sample rate, not the spectrogram framerate.
                           "framerate": 1 / 0.010,
                           "fmin": 20,
                           "fmax": 20000,
                           "n_fft": 2048,
                           "n_mels": 300})
    print("")
