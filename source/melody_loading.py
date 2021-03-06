import pandas as pd
from mido import MidiFile
import mido
import librosa
import numpy as np
import math
from data_loading import *
import scipy.io.wavfile


signal_Fs = 44100
midi_dir = "data/midi/sebasgverde_mono-midi-transposition/train"                            # Input midi files directory
melody_data_dir = os.path.join(pickled_data_dir, "melody_dataset", "melody_dataset_final")  # Output pickle directory
no_melodies = 20   # Parameter to set how many midi files to use in dataset per velocity layer per instrument
envelope_plotting = False


class SignalWriter:
    def __init__(self, instrument, fs, duration):
        self.instrument = instrument
        self.Fs = fs
        self.waveform = np.zeros(shape=math.ceil(self.Fs * duration), dtype="int32")

    def add_note(self, start_time, end_time, pitch):
        if end_time - start_time < 0.08:
            # Filter out short notes which cause glitchy sounds
            return
        # print("Writing note to waveform from", start_time, "to", end_time, "at pitch", pitch)
        start_index = int(start_time * self.Fs)
        end_index = int(end_time * self.Fs)
        note_len = end_index-start_index
        sampled_note_row = self.instrument.loc[self.instrument.pitch == pitch]
        if sampled_note_row.empty:
            raise Exception("Pitch "+str(pitch)+" doesn't exist in the samples for the current instrument")
        elif sampled_note_row.shape[0] > 1:
            # Pick a random sample for the note to increase dataset variety when multiple samples exist
            rand_index = np.random.randint(sampled_note_row.shape[0])
            # print("Found multiple samples matching pitch", pitch, "for the current instrument, selecting sample index", rand_index)
            sampled_note = (sampled_note_row.iloc[rand_index].copy()).waveform.copy()
        else:
            sampled_note = (sampled_note_row.iloc[0].copy()).waveform.copy()

        # Convert from float32 to int32 if necessary
        if sampled_note.dtype == np.float32:
            sampled_note = (sampled_note*2147483647/np.max(np.abs(sampled_note))).astype(np.int32)

        if note_len > len(sampled_note):
            # print("Duration", note_len, "is longer than sample")
            sampled_note = np.pad(sampled_note, (0, note_len-len(sampled_note)))
        else:
            # Cut sample to note length
            sampled_note = sampled_note[:note_len]

        # Apply an attack/release envelope to the note: concave attack and release to remove clicks
        attack_time = 0.005     # 5 ms attack time
        attack_len = int(self.Fs*attack_time)
        fade_in = np.sqrt(np.linspace(start=0, stop=1, num=attack_len))
        if envelope_plotting:
            # Before and after envelope shaping plots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8), sharey="col")
            fig.suptitle("Single-note waveform attack and release, before and after envelope-shaping is applied", size='x-large')
            # "Before" attack plot
            ax1.plot(sampled_note[:int(attack_len/2)]/np.max(np.abs(sampled_note)))
            ax1.set_title("Before attack-shaping", style='italic')
            y_min, y_max = ax1.get_ylim()
            ax1.set_ylim(y_min, -y_min)
            ax1.grid(axis="y")
        sampled_note[:attack_len] = (sampled_note[:attack_len] * fade_in).astype("int32")

        release_time = 0.005    # 5 ms release time
        release_len = int(self.Fs*release_time)
        fade_out = np.sqrt(np.linspace(start=1, stop=0, num=release_len))
        if envelope_plotting:
            # "Before" release plot
            ax2.plot(sampled_note[-release_len:]/np.max(np.abs(sampled_note)))
            ax2.set_title("Before release-shaping", style='italic')
            ax2.set_xticks(ticks=[0, 50, 100, 150, 200])
            ax2.set_xticklabels(labels=np.array([0, 50, 100, 150, 200]) - release_len)
            ax2.grid(axis="y")
        sampled_note[-release_len:] = (sampled_note[-release_len:] * fade_out).astype("int32")

        if envelope_plotting:
            # "After" plots
            ax3.plot(sampled_note[:int(attack_len/2)]/np.max(np.abs(sampled_note)))
            ax3.set(xlabel="Sample index (from waveform start)")
            ax3.set_title("After attack-shaping", style='italic')
            ax3.grid(axis="y")
            ax4.plot(sampled_note[-release_len:]/np.max(np.abs(sampled_note)))
            ax4.set_xticks(ticks=[0, 50, 100, 150, 200])
            ax4.set_xticklabels(labels=np.array([0, 50, 100, 150, 200])-release_len)
            ax4.set(xlabel="Sample index (from waveform end)")
            ax4.set_title("After release-shaping", style='italic')
            ax4.grid(axis="y")
            fig.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
            plt.ylabel("Normalised Amplitude", labelpad=20, size='large')
            plt.show()
            fig.savefig("../Figures/Waveform_envelopes.svg")

            # Envelope shaping attack/release function plots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
            ax1.plot(np.arange(attack_len), fade_in)
            ax1.set(xlabel="Sample index "+r'n'+" (from waveform start)", ylabel="Gain "+r'y')
            ax1.set_title("Attack-shaping fade-in function ("+r'y=$\sqrt{n}$'+")")
            ax2.plot(np.arange(release_len), fade_out)
            ax2.set(xlabel="Sample index "+r'-n'+" (from waveform end)")
            ax2.set_xticks(ticks=[0, 50, 100, 150, 200])
            ax2.set_xticklabels(labels=np.array([0, 50, 100, 150, 200])-release_len)
            ax2.set_title("Release-shaping fade-out function ("+r'y=$\sqrt{end-n}$'+")")
            plt.tight_layout()
            plt.show()
            fig.savefig("../Figures/Envelope_functions.svg")

        self.waveform[start_index:end_index] = sampled_note
        return


class MelodyInstrumentLoader(InstrumentLoader):
    def __init__(self, data_dir, note_range=None, set_velocity=None, normalise_wavs=True, load_MIDIsampled=True, reload_wavs=False):
        super().__init__(data_dir, note_range, set_velocity, normalise_wavs, load_MIDIsampled, reload_wavs)
        self.avg_note_lengths = []

    def sequence_melody(self, midi_path, sample_instrument):
        mid = MidiFile(midi_path)

        print("Type:", mid.type)
        if mid.type == 2:
            raise Exception("Unsupported midi type:", mid.type)
        if len(mid.tracks) != 1:
            raise Exception("Unsupported number of tracks:", len(mid.tracks))

        midi_track = mid.tracks[0]

        # Ensure the melody fits into instrument's range, transpose if necessary
        note_max = max(msg.note for msg in mid.tracks[0] if msg.type == "note_on")
        note_min = min(msg.note for msg in mid.tracks[0] if msg.type == "note_on")
        instrument_max = max(sample_instrument.pitch)
        instrument_min = min(sample_instrument.pitch)

        if note_max-note_min < 3:
            raise Exception("Melody's note range is too small")

        if note_max-note_min > instrument_max-instrument_min:
            raise Exception("Melody's note range is too large for the given instrument")
        transposition = instrument_max-note_max

        tempo = 500000
        try:
            length = mid.length
        except Exception as exception:
            print("Length could not be determined, assuming tempo of 120 BPM")
            length = mido.tick2second(sum(msg.time for msg in mid.tracks[0]), mid.ticks_per_beat, tempo)
        print("Length:", length)

        signal = SignalWriter(sample_instrument, signal_Fs, length)

        current_time = 0
        current_note = 0
        current_vel = 0
        note_active = False
        note_counter = 0
        for midi_msg in midi_track:
            #print(midi_msg)
            # Read tempo from header meta messages
            if midi_msg.type == "set_tempo":
                if tempo == 500000:
                    tempo = midi_msg.tempo
                else:
                    raise Exception("Multiple tempos read")

            # Read note on and off messages
            if midi_msg.type == "note_on" or midi_msg.type == "note_off":
                # Compute note timestamp
                time_delta = mido.tick2second(midi_msg.time, mid.ticks_per_beat, tempo)
                if time_delta > 4 and current_time != 0:
                    # Skip melodies with long pauses between notes
                    raise Exception("Long pause of "+str(time_delta)+" s detected")
                current_time += time_delta
                #print("Current time:", current_time)
                if midi_msg.type == "note_on":
                    if note_active:
                        # Add the current note to the waveform before registering a new note_on
                        signal.add_note(start_time, current_time, current_note+transposition)
                        note_counter += 1
                    note_active = True
                    current_note = midi_msg.note
                    current_vel = midi_msg.velocity
                    start_time = current_time
                if midi_msg.type == "note_off":
                    if not note_active:
                        print("Ignoring note_off received without active note")
                        continue
                    if midi_msg.note != current_note:
                        raise Exception("note_off pitch didn't match previous note_on")
                    note_active = False
                    end_time = current_time
                    signal.add_note(start_time, end_time, current_note+transposition)
                    note_counter += 1

        # Trim zeros to remove any silences at the start or end of the melody
        waveform = np.trim_zeros(signal.waveform)
        self.avg_note_lengths.append((len(waveform)/signal.Fs)/note_counter)
        return waveform

    def preprocess_melodies(self, midi_dir, normalisation=None):
        out = []
        i = 0
        # If single-note wavs don't exist, we assume pre-processed pickles are available in melody_data_dir
        instrument_list = [os.path.splitext(f)[0] for f in os.listdir(melody_data_dir)] if self.dataset.empty \
            else pd.unique(self.dataset.instrument)
        for instrument_name in instrument_list:
            instrument_pkl_path = os.path.join(melody_data_dir, instrument_name+".pkl")
            if not os.path.isfile(instrument_pkl_path):
                print("\n\n", instrument_pkl_path, "not found, applying melody generation to", instrument_name)
                instrument_samples = self.dataset.loc[self.dataset.instrument == instrument_name]
                instrument_out = pd.DataFrame()

                for velocity in pd.unique(instrument_samples.velocity):
                    velocity_samples = instrument_samples.loc[instrument_samples.velocity == velocity]
                    total_melodies = 0
                    while total_melodies < no_melodies:
                        if i >= len(os.listdir(midi_dir)):
                            raise Exception("Ran out of midi files to read in directory", midi_dir, "at index", i)

                        melody_midi = os.listdir(midi_dir)[i]
                        melody_mid_path = os.path.join(midi_dir, melody_midi)
                        print("\nApplying melody", melody_midi, "to instrument", instrument_name, "velocity", velocity)

                        try:
                            melody_waveform = self.sequence_melody(melody_mid_path, velocity_samples)
                            # soundfile.write("data/test_" + melody_midi + "_" + instrument_name + "_" + velocity + ".wav",
                            #                 (2147483647*(melody_waveform/np.max(np.abs(melody_waveform)))).astype(np.int32), signal_Fs)

                            melody_melspec = self.compute_spectrogram(melody_waveform, signal_Fs)
                            melspec_frames = sample_frames(melody_melspec, frame_len=221)
                            for melspec in melspec_frames:
                                # Normalise each spectrogram's magnitudes
                                if normalisation == "statistics":
                                    melspec = (melspec - np.mean(melspec)) / np.std(melspec)

                                spec_row = pd.DataFrame({"dataset": velocity_samples.iloc[0]["dataset"],
                                                         "instrument": instrument_name,
                                                         "spectrogram": [melspec],
                                                         "melody": melody_midi,
                                                         "velocity": velocity,
                                                         "label": velocity_samples.iloc[0]["label"]})
                                instrument_out = instrument_out.append(spec_row)
                            total_melodies += 1     # Count the number of correctly sequenced and processed melodies
                        except Exception as e:
                            print("Skipping melody", melody_mid_path, "due to sequencing error:")
                            print("\t", e)
                        # Iterate through the files in midi directory so that different melodies are used
                        # for each velocity layer and each instrument
                        i += 1
                    # Print note length statistics once the set number of melodies has been generated
                    average_note_len = np.mean(self.avg_note_lengths)
                    max_avg_note_len = np.max(self.avg_note_lengths)
                    print("\nMelodies contain notes of average length", average_note_len, "s")
                    print("\t yielding", 2.21/average_note_len, "notes per 2.21 s window")
                    print("The longest average note length per melody is", max_avg_note_len, "s")
                    print("\t yielding", 2.21/max_avg_note_len, "notes per 2.21 s window\n")
                    self.avg_note_lengths = []
                instrument_out.to_pickle(instrument_pkl_path)
                print("Saved pickle to", instrument_pkl_path)
            else:
                print("Loading pickle from", instrument_pkl_path)
                instrument_out = pd.read_pickle(instrument_pkl_path)

            # Remove spectrograms of incorrect shape
            instrument_out = instrument_out[instrument_out["spectrogram"].map(np.shape) == (300, 221)]

            # Remove any spectrograms that contain NaNs
            instrument_out = instrument_out[instrument_out["spectrogram"].map(lambda spec: np.isfinite(spec).all())]
            # Convert instrument's label to binary, "Grand" = 0, "Upright" = 1
            instrument_out.label = instrument_out.label.replace("Grand", 0)
            instrument_out.label = instrument_out.label.replace("Upright", 1)

            out.append(instrument_out)
        return pd.concat(out)

    def stack_context(self):
        return


# Helper function for sampling non-overlapping windows of a desired length from a waveform or spectrogram
def sample_frames(signal, frame_len):
    if signal.ndim == 1:
        frames = [signal[x:x + frame_len] for x in range(0, len(signal), frame_len)]
    elif signal.ndim == 2:
        frames = [signal[:, x:x + frame_len] for x in range(0, signal.shape[1], frame_len)]
    else:
        raise Exception("Cannot sample frames from array of dimension > 2")
    return frames


if __name__ == '__main__':
    # melody_loader = MelodyInstrumentLoader(data_dir, note_range=[48, 72], set_velocity=None, normalise_wavs=True)
    # melody_melspec_data = melody_loader.preprocess_melodies(midi_dir, normalisation="statistics")
    print("")

