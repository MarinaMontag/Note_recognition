import math
import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import scipy.signal
import sounddevice as sd
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq
from scipy.signal import savgol_filter
from music21.pitch import Pitch
from midiutil.MidiFile import MIDIFile
from spleeter.separator import Separator


def plot_signal_from_file(filename):
    sr, y = wavfile.read(filename)
    plt.plot(y)
    plt.show()
    # sd.play(y, sr)


def smooth_signal(data, times):
    data = data[:, 0] + data[:, 1]
    b, a = scipy.signal.butter(3, .1)
    filteredGust = scipy.signal.filtfilt(b, a, data, method="gust")
    N = 100
    e = 0.1
    diff_std = 1000
    while True:
        # data = np.abs(data)
        filtered = savgol_filter(filteredGust, N, 3)
        peaks, _ = find_peaks(filtered, distance=6000, height=2000)
        peaks = np.append(0, peaks)
        peaks = np.append(peaks, len(data) - 1)
        diff = [j - i for i, j in zip(peaks[:-1], peaks[1:])]


        if True:
            break

        # if np.std(diff) - diff_std > 0:
        #     N -= 1
        #     break
        # if np.std(diff) < 0:
        #     break
        # diff_std = np.std(diff)
        # N += 1

    plt.figure(figsize=(10, 4))

    plt.subplot(121)
    plt.plot(times, data)
    plt.title("Non-filtered Signal")
    plt.margins(0, .05)

    plt.subplot(122)
    plt.plot(times, filtered)
    plt.title("Filtered ECG Signal")
    plt.margins(0, .05)

    plt.tight_layout()
    plt.show()
    return filtered, peaks


def plot_peaks(data, peaks):
    plt.plot(data)
    plt.plot(peaks, data[peaks], "x")
    plt.plot(np.zeros_like(data), "--", color="gray")
    plt.show()
    return peaks


def get_dominate_frequencies(filename, peaks):
    sr, vocals = wavfile.read(filename)
    T = 1.0 / sr
    dom_tones = []
    E2 = 82
    G7 = 3136
    needed_peaks = []
    for i in range(len(peaks) - 1):
        yf = fft(vocals[peaks[i]:peaks[i + 1], 0], peaks[i + 1] - peaks[i])[:(peaks[i + 1] - peaks[i]) // 2] + \
             fft(vocals[peaks[i]:peaks[i + 1], 1], peaks[i + 1] - peaks[i])[:(peaks[i + 1] - peaks[i]) // 2]
        xf = fftfreq(peaks[i + 1] - peaks[i], T)[:(peaks[i + 1] - peaks[i]) // 2]
        energy = np.abs(yf)
        dom = xf[np.argmax(energy)]
        if E2 <= dom <= G7:
            dom_tones.append(dom)
            needed_peaks.append(i)
    return dom_tones, needed_peaks


def freq_to_note(freq):
    notes = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']

    note_number = 12 * math.log2(freq / 440) + 49
    note_number = round(note_number)

    note = (note_number - 1) % len(notes)
    note = notes[note]

    octave = (note_number + 8) // len(notes)

    return note, octave


def get_musical_notes(frequencies):
    notes = []
    for freq in frequencies:
        notes.append(freq_to_note(freq))
    for i in range(len(notes)):
        note = ''.join(str(j) for j in notes[i])
        notes[i] = note
    return notes

def get_signal(dom_freq, needed_peaks, peaks, sampleRate):
    sig = np.array([])
    j = 0
    for i in needed_peaks:
        t = np.arange(peaks[i], peaks[i + 1]) / sampleRate
        sig = np.append(sig, np.sin(2 * np.pi * dom_freq[j] * t))
        j += 1
    return sig

def play_sound(sig, sr):
    sd.play(sig, sr)
    sd.wait()

def create_midi(notes):
    mf = MIDIFile(1)  # only 1 track
    track = 0  # the only track
    mf.addTrackName(track, 0, "Sample Track")
    mf.addTempo(track, 0, 120)

    # add some notes
    channel = 0
    volume = 100
    duration = 1  # 1 beat long
    for i in range(len(notes)):
        pitch = int(Pitch(notes[i]).ps)
        time = i
        mf.addNote(track, channel, pitch, time, duration, volume)

    # write it to disk
    with open("output.mid", 'wb') as outf:
        mf.writeFile(outf)

def set_drums_vocals(song_path, output_directory_path):
    separator = Separator('spleeter:4stems')
    separator.separate_to_file(song_path, output_directory_path)

def get_sources_directory_path(song_path):
    folders = song_path.split('\\')
    song_name_part = folders[-1]
    song_name_part = song_name_part.split('.')
    song_name_part.pop()
    song_name_part = '.'.join(song_name_part)
    return 'audio\\'+song_name_part

def main(song_file):
    set_drums_vocals(song_file, "audio")
    sources_path = get_sources_directory_path(song_file)
    drums = sources_path + '\\' + 'drums.wav'
    vocals = sources_path + '\\' + 'vocals.wav'
    sample_rate, data = wavfile.read(drums)
    times = np.arange(0, len(data)) / sample_rate
    filtered, peaks = smooth_signal(data, times)
    plot_peaks(filtered, peaks)
    dom_freq, needed_peaks = get_dominate_frequencies(vocals, peaks)
    notes = get_musical_notes(dom_freq)
    print(notes)
    create_midi(notes)


if __name__ == '__main__':
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    parser = argparse.ArgumentParser()
    parser.add_argument("song_file")
    args = parser.parse_args()
    main(args.song_file)










