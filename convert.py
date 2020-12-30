from music21 import converter, note, chord, stream
import numpy as np
import os
import pandas as pd
import csv


# Melody-RNN Format is a sequence of 8-bit integers indicating the following:
# MELODY_NOTE_ON = [0, 127] # (note on at that MIDI pitch)
MELODY_NOTE_OFF = 128  # (stop playing all previous notes)
MELODY_NO_EVENT = 129  # (no change from previous event)


# Each element in the sequence lasts for one sixteenth note.
# This can encode monophonic music only.

def stream_to_note_array(stream):
    """
    Convert a Music21 sequence to a numpy array of int8s into Melody-RNN format:
        0-127 - note on at specified pitch
        128   - note off
        129   - no event
    """
    # Part one, extract from stream
    total_length = np.int(np.round(stream.flat.highestTime / 0.25))  # in semiquavers
    stream_list = []
    for element in stream.flat:
        if isinstance(element, note.Note):
            stream_list.append(
                [np.round(element.offset / 0.25), np.round(element.quarterLength / 0.25), element.pitch.midi])
        elif isinstance(element, chord.Chord):
            stream_list.append([np.round(element.offset / 0.25), np.round(element.quarterLength / 0.25),
                                element.sortAscending().pitches[-1].midi])
    np_stream_list = np.array(stream_list, dtype=np.int)
    df = pd.DataFrame({'pos': np_stream_list.T[0], 'dur': np_stream_list.T[1], 'pitch': np_stream_list.T[2]})
    df = df.sort_values(['pos', 'pitch'], ascending=[True, False])  # sort the dataframe properly
    df = df.drop_duplicates(subset=['pos'])  # drop duplicate values
    # part 2, convert into a sequence of note events
    output = np.zeros(total_length + 2, dtype=np.int16) + np.int16(
        MELODY_NO_EVENT)  # set array full of no events by default.
    # Fill in the output list
    for i in range(total_length):
        if not df[df.pos == i].empty:
            n = df[df.pos == i].iloc[0]  # pick the highest pitch at each semiquaver
            output[i] = n.pitch  # set note on
            output[i + n.dur] = MELODY_NOTE_OFF
    return output


def note_array_to_dataframe(note_array):
    """
    Convert a numpy array containing a Melody-RNN sequence into a dataframe.
    """
    df = pd.DataFrame({"code": note_array})
    df['offset'] = df.index
    df['duration'] = df.index
    df = df[df.code != MELODY_NO_EVENT]
    df.duration = df.duration.diff(-1) * -1 * 0.25  # calculate durations and change to quarter note fractions
    df = df.fillna(0.25)
    return df[['code', 'duration']]


def note_array_to_stream(note_array):
    """
    Convert a numpy array containing a Melody-RNN sequence into a music21 stream.
    """
    df = note_array_to_dataframe(note_array)
    melody_stream = stream.Stream()
    for index, row in df.iterrows():
        if row.code == MELODY_NO_EVENT:
            new_note = note.Rest()  # bit of an oversimplification, doesn't produce long notes.
        elif row.code == MELODY_NOTE_OFF:
            new_note = note.Rest()
        else:
            new_note = note.Note(row.code)
        new_note.quarterLength = row.duration
        melody_stream.append(new_note)
    return melody_stream


def clean(sequence):
    while True:
        i = len(sequence) - 1
        if sequence[i] == MELODY_NOTE_OFF or sequence[i] == MELODY_NO_EVENT:
            del (sequence[i])
        else:
            break
    sequence.append(MELODY_NOTE_OFF)

    while True:
        if sequence[0] == MELODY_NO_EVENT:
            del (sequence[0])
        else:
            break
    return sequence


def midi_to_txt(path, output_dir):
    wm_mid = converter.parse(path)
    # wm_mid.show()
    wm_mel_rnn = stream_to_note_array(wm_mid)
    text = wm_mel_rnn.tolist()
    text = clean(text)
    file_name = os.path.split(path)[-1]
    save_name = file_name.replace('midi', 'txt')
    save_path = os.path.join('.', output_dir, save_name)
    with open(save_path, 'w') as f:
        for i in range(len(text)):
            f.write(str(text[i]))
            if i != len(text) - 1:
                f.write(' ')


def txt_to_midi(path, output_dir):
    with open(path, 'r') as f:
        text = f.read().strip().split(' ')
        data = []
        for item in text:
            data.append(int(item))
    s = note_array_to_stream(data)
    save_name = os.path.split(path)[-1].replace('txt', 'midi')
    out_path = os.path.join('.', output_dir, save_name)
    s.write('midi', fp=out_path)


def get_path(composer):
    with open('maestro-v3.0.0.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        reader = list(reader)
    del (reader[0])
    files = []
    for line in reader:
        if line[0] == composer:
            files.append(line[4])
    paths = []
    for file in files:
        path = os.path.join('.', 'midi_data', file)
        path = path.replace('/', '\\')
        paths.append(path)
    return paths


if __name__ == '__main__':
    # composers = ['Franz Schubert', 'Johann Sebastian Bach']
    # for composer in composers:
    #     if os.path.exists(composer):
    #         print('文件夹{}已存在'.format(composer))
    #     else:
    #         os.makedirs(composer)
    #         print('创建文件夹{}'.format(composer))
    #     paths = get_path(composer)
    #
    #     tmp_files = os.listdir(composer)
    #     done_files = []
    #     for file in tmp_files:
    #         file = file.split('\\')[-1].replace('txt', 'midi')
    #         done_files.append(file)
    #
    #     i = 1
    #     for path in paths:
    #         if path.split('\\')[-1] not in done_files:
    #             midi_to_txt(path, composer)
    #         print('{}/{}'.format(i, len(paths)))
    #         i += 1

    txt_to_midi(path='versions/tmp/hide/bin.txt', output_dir='versions/midi')
    # midi_to_txt(path='versions/midi/result2.midi', output_dir='versions')