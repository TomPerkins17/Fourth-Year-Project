from mido import MidiFile
import mido
import librosa
import numpy as np
import math
from data_loading import *
import scipy.io.wavfile

Fs = 44100
midi_dir = "data/midi/sebasgverde_mono-midi-transposition/validation"   # Input midi files directory
melody_data_dir = "data/melody_dataset_normalisedwavs_20"                  # Output pickle files directory
no_melodies = 20   # Parameter to set how many midi files to use in dataset


class SignalWriter:
    def __init__(self, instrument, Fs, duration):
        self.instrument = instrument
        self.Fs = Fs
        self.waveform = np.zeros(shape=math.ceil(Fs * duration), dtype="int32")

    def add_note(self, start_time, end_time, pitch, debug=False):
        if end_time - start_time < 0.08:
            # Filter out  short notes which cause glitchy sounds
            return
        #print("Writing note to waveform from", start_time, "to", end_time, "at pitch", pitch)
        start_index = int(start_time * self.Fs)
        end_index = int(end_time * self.Fs)
        note_len = end_index-start_index
        sampled_note_row = self.instrument.loc[self.instrument.pitch == pitch]
        if sampled_note_row.empty:
            print("Pitch", pitch, "doesn't exist in the samples for the current instrument")
            return
        elif sampled_note_row.shape[0] > 1:
            print("Found multiple samples matching pitch", pitch, "for the current instrument")
        sampled_note = (sampled_note_row.iloc[0].copy()).waveform.copy()
        if note_len > len(sampled_note):
            print("Duration", note_len, "is longer than sample")
            sampled_note = np.pad(sampled_note, (0, note_len-len(sampled_note)))
        else:
            # Cut sample to note length
            sampled_note = sampled_note[:note_len]

        if debug:
            plt.plot(sampled_note)
            plt.show()

        # Apply an attack/delay envelope to the note: concave attack and release to remove clicks
        attack_time = 0.005     # 5 ms attack time
        attack_len = int(self.Fs*attack_time)
        fade_in = np.sqrt(np.linspace(start=0, stop=1, num=attack_len))
        sampled_note[:attack_len] = (sampled_note[:attack_len] * fade_in).astype("int32")

        release_time = 0.005    # 5 ms release time
        release_len = int(self.Fs*release_time)
        fade_out = np.sqrt(np.linspace(start=1, stop=0, num=release_len))
        sampled_note[-release_len:] = (sampled_note[-release_len:] * fade_out).astype("int32")

        if debug:
            plt.plot(sampled_note)
            plt.show()

        self.waveform[start_index:end_index] = sampled_note
        return


class MelodyInstrumentLoader(InstrumentLoader):
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

        if note_max-note_min > instrument_max-instrument_min:
            raise Exception("Melody's note range is too large for the given instrument")
        transposition = instrument_max-note_max

        tempo = 500000
        try:
            length = mid.length
        except Exception as exception:
            print("Length could not be determined, assuming tempo of 120 BPM, giving:")
            length = mido.tick2second(sum(msg.time for msg in mid.tracks[0]), mid.ticks_per_beat, tempo)
        print("Length:", length)

        signal = SignalWriter(sample_instrument, Fs, length)

        current_time = 0
        current_note = 0
        current_vel = 0
        note_active = False

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
                current_time += time_delta
                #print("Current time:", current_time)
                if midi_msg.type == "note_on":
                    if note_active:
                        # Add the current note to the waveform before registering a new note_on
                        signal.add_note(start_time, current_time, current_note+transposition)
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

        return signal.waveform

    def preprocess_melodies(self, midi_dir):
        out = []
        for instrument_name in pd.unique(self.dataset.instrument):
            instrument_pkl_path = os.path.join(melody_data_dir, instrument_name+".pkl")
            if not os.path.isfile(instrument_pkl_path):
                instrument_samples = self.dataset.loc[self.dataset.instrument == instrument_name]
                instrument_out = pd.DataFrame()

                for velocity in pd.unique(instrument_samples.velocity):
                    velocity_samples = instrument_samples.loc[instrument_samples.velocity == velocity]
                    total_melodies = 0
                    for melody_midi in os.listdir(midi_dir):
                        melody_mid_path = os.path.join(midi_dir, melody_midi)
                        print("Applying melody", melody_mid_path, "to instrument", instrument_name, "velocity", velocity)
                        try:
                            melody_waveform = self.sequence_melody(melody_mid_path, velocity_samples)
                            # soundfile.write("data/test_"+melody_midi+"_"+instrument_name+"_"+velocity+".wav",
                            #                 (melody_waveform.astype(np.float32)/np.max(np.abs(melody_waveform)).astype(np.float32)), Fs)
                            melody_melspec = self.compute_spectrogram(melody_waveform, Fs)
                            melspec_frames = sample_frames(melody_melspec, frame_len=221)
                            for melspec in melspec_frames:
                                spec_row = pd.DataFrame({"dataset": velocity_samples.iloc[0]["dataset"],
                                                         "instrument": instrument_name,
                                                         "spectrogram": [melspec],
                                                         "melody": melody_midi,
                                                         "velocity": velocity,
                                                         "label": velocity_samples.iloc[0]["label"]})
                                instrument_out = instrument_out.append(spec_row)
                        except Exception as e:
                            print(e)
                        total_melodies += 1
                        if total_melodies > no_melodies:
                            break
                instrument_out.to_pickle(instrument_pkl_path)
            else:
                print("Loading pickle from", instrument_pkl_path)
                instrument_out = pd.read_pickle(instrument_pkl_path)

            # TODO: spectrogram post processing (normalisation)

            # Convert instrument's label to binary, "Grand" = 0, "Upright" = 1
            instrument_out.label = instrument_out.label.replace("Grand", 0)
            instrument_out.label = instrument_out.label.replace("Upright", 1)

            # Remove spectrograms of incorrect shape
            instrument_out = instrument_out[instrument_out["spectrogram"].map(np.shape) == (300, 221)]
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
    melody_loader = MelodyInstrumentLoader(data_dir, note_range=[48, 72], set_velocity=None, normalise_wavs=True)
    melody_melspec_data = melody_loader.preprocess_melodies(midi_dir)

    # print("")

