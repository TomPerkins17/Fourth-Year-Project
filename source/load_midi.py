from mido import MidiFile
import mido
import librosa
import numpy as np
import math
from data_loading import *
import scipy.io.wavfile

Fs = 44100
midi_path = 'data/trip_to_pakistan.mid'


class SignalWriter:
    def __init__(self, instrument, Fs, duration):
        self.instrument = instrument
        self.Fs = Fs
        self.waveform = np.zeros(shape=math.ceil(Fs * duration), dtype="int32")

    def add_note(self, start_time, end_time, pitch, debug=False):
        print("Writing note to waveform from", start_time, "to", end_time, "at pitch", pitch)
        if pitch > self.instrument.pitch.max() or pitch < self.instrument.pitch.min():
            print("Pitch", pitch, "is outside of sample instrument's range")
            return
        start_index = int(start_time * self.Fs)
        end_index = int(end_time * self.Fs)
        note_len = end_index-start_index
        sampled_note_row = self.instrument.loc[self.instrument.pitch == pitch]
        sampled_note = (sampled_note_row.copy()).waveform[0].copy()

        if note_len > len(sampled_note):
            print("Duration", note_len, "is longer than sample")
            return

        # Cut sample to note length
        sampled_note = sampled_note[:note_len]
        if debug:
            plt.plot(sampled_note)
            plt.show()
            print("")

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
            print("")

        self.waveform[start_index:end_index] = sampled_note
        return


def sequence_melody(midi_path, sample_instrument):
    mid = MidiFile(midi_path)
    signal = SignalWriter(sample_instrument, Fs, mid.length)

    print("Length:", mid.length)
    print("Type:", mid.type)
    if len(mid.tracks) != 1:
        raise Exception("Unsupported number of tracks:", len(mid.tracks))
    midi_track = mid.tracks[0]
    print('Track {}: {}'.format(0, midi_track.name))

    current_time = 0
    current_note = 0
    current_vel = 0
    note_active = False
    tempo = 0

    note_counter = 0
    for midi_msg in midi_track:
        print(midi_msg)
        # Read tempo from header meta messages
        if midi_msg.type == "set_tempo":
            if not tempo:
                tempo = midi_msg.tempo
            else:
                raise Exception("Multiple tempos read")
        # Read note on and off messages
        if midi_msg.type == "note_on" or midi_msg.type == "note_off":
            # Compute note timestamp
            if not tempo:
                raise Exception("No tempo specified")
            else:
                time_delta = mido.tick2second(midi_msg.time, mid.ticks_per_beat, tempo)
                current_time += time_delta
                print("Current time:", current_time)
            if midi_msg.type == "note_on":
                if note_active:
                    raise Exception("note_on received while note already active")
                note_active = True
                current_note = midi_msg.note
                current_vel = midi_msg.velocity
                start_time = current_time
            if midi_msg.type == "note_off":
                if not note_active:
                    raise Exception("note_off received without active note")
                if midi_msg.note != current_note:
                    raise Exception("note_off pitch didn't match previous note_on")
                note_active = False
                end_time = current_time
                note_counter += 1
                signal.add_note(start_time, end_time, current_note)

    return signal.waveform


def sample_frames(wave, frame_len):
    frames = [wave[x:x + frame_len] for x in range(0, len(wave), frame_len)]
    return frames


if __name__ == '__main__':
    loader = InstrumentLoader(data_dir, note_range=[48, 72], set_velocity="M")
    instrument = loader.dataset.loc[loader.dataset.instrument == "AkPnBcht"]

    # mid = MidiFile(midi_path)
    # signal = SignalWriter(instrument, Fs, mid.length)
    # signal.add_note(0.25, 0.498958333333334, 67, debug=True)
    # signal.add_note(1.0, 1.7489583333333334, 67, debug=True)

    wave = sequence_melody(midi_path, instrument)
    #soundfile.write("data/test_envelope.wav", (wave.astype(np.float32)/np.max(np.abs(wave)).astype(np.float32)), Fs)
    frames = sample_frames(wave, frame_len=97285) # frame len of about 2.21s chosen to match single-note length
    for frame in frames:
        melspec = compute_spectrogram(frame, Fs)
    print("")
