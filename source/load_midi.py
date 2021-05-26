from mido import MidiFile
import mido
import librosa
import numpy as np
import math
from data_loading import *
import scipy.io.wavfile

Fs = 44100
midi_path = 'data/midi/trip_to_pakistan.mid'
midi_dir = "data/midi/sebasgverde_mono-midi-transposition/validation"


class SignalWriter:
    def __init__(self, instrument, Fs, duration):
        self.instrument = instrument
        self.Fs = Fs
        self.waveform = np.zeros(shape=math.ceil(Fs * duration), dtype="int32")

    def add_note(self, start_time, end_time, pitch, debug=False):
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
            return

        # Cut sample to note length
        sampled_note = sampled_note[:note_len]
        if debug:
            plt.plot(sampled_note)
            plt.show()

        # Apply an attack/delay envelope to the note: concave attack and release to remove clicks
        attack_time = 0.005     # 5 ms attack time
        attack_len = int(self.Fs*attack_time)
        if 0.5*note_len < attack_len:
            print("Note too short to apply envelope, skipping")
            return
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


def sequence_melody(midi_path, sample_instrument):
    mid = MidiFile(midi_path)

    print("Type:", mid.type)
    if mid.type == 2:
        raise Exception("Unsupported midi type:", mid.type)
    if len(mid.tracks) != 1:
        raise Exception("Unsupported number of tracks:", len(mid.tracks))

    midi_track = mid.tracks[0]
    tempo = 500000
    try:
        print("Length:", mid.length)
        length = mid.length
    except Exception as exception:
        print(exception)
        print("Assuming tempo of 120 BPM")
        length = mido.tick2second(sum(msg.time for msg in mid.tracks[0]), mid.ticks_per_beat, tempo)

    print('Track {}: {}'.format(0, midi_track.name))

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
                    signal.add_note(start_time, current_time, current_note)
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
                signal.add_note(start_time, end_time, current_note)

    return signal.waveform


def sample_frames(wave, frame_len):
    if wave.ndim == 1:
        frames = [wave[x:x + frame_len] for x in range(0, len(wave), frame_len)]
    elif wave.ndim == 2:
        frames = [wave[:, x:x + frame_len] for x in range(0, wave.shape[1], frame_len)]
    else:
        raise Exception("Cannot sample frames from array of dimension > 2")
    return frames


def load_melody_dataset(instrument_dataset, midi_dir):
    out = pd.DataFrame()
    for melody_midi in os.listdir(midi_dir):
        melody_mid_path = os.path.join(midi_dir, melody_midi)
        for instrument_name in pd.unique(instrument_dataset.instrument):
            print("Applying melody", melody_mid_path, "to instrument", instrument_name)
            instrument_samples = instrument_dataset.loc[instrument_dataset.instrument == instrument_name]

            melody_waveform = sequence_melody(melody_mid_path, instrument_samples)
            #soundfile.write("data/test_envelope.wav", (melody_waveform.astype(np.float32)/np.max(np.abs(melody_waveform)).astype(np.float32)), Fs)

            # melody_frames = sample_frames(melody_waveform, frame_len=97285) # frame len of about 2.21s chosen to match single-note length
            # for frame in melody_frames:
            #     melspec = compute_spectrogram(frame, Fs, plot=True)
            #     spec_row = pd.DataFrame({"dataset": instrument_samples.iloc[0]["dataset"],
            #                              "instrument": instrument_name,
            #                              "spectrogram": [melspec],
            #                              "melody": midi_path,
            #                              "label": instrument_samples.iloc[0]["label"]})
            #     out = out.append(spec_row)

            melody_melspec = compute_spectrogram(melody_waveform, Fs)
            melspec_frames = sample_frames(melody_melspec, frame_len=221)
            for melspec in melspec_frames:
                spec_row = pd.DataFrame({"dataset": instrument_samples.iloc[0]["dataset"],
                                         "instrument": instrument_name,
                                         "spectrogram": [melspec],
                                         "melody": melody_midi,
                                         "label": instrument_samples.iloc[0]["label"]})
                out = out.append(spec_row)
            print("")
    return out


if __name__ == '__main__':
    loader = InstrumentLoader(data_dir, note_range=[48, 72], set_velocity="M")
    melody_melspec_data = load_melody_dataset(loader.dataset, midi_dir)

    # mid = MidiFile(midi_path)
    # signal = SignalWriter(instrument, Fs, mid.length)
    # signal.add_note(0.25, 0.498958333333334, 67, debug=True)
    # signal.add_note(1.0, 1.7489583333333334, 67, debug=True)
