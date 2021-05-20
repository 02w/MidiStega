import csv
import os
from functools import reduce
from operator import add

import muspy
import numpy as np


def midi_to_events(file, mode='note', longest_track_only=True):
    """
    Convert a midi file to event-based representation.
    """
    try:
        music = muspy.read_midi(file)
        resolution = music.resolution
        if longest_track_only:
            # choose the longest track
            track_len = [len(i) for i in music.tracks]
            track = music.tracks[track_len.index(max(track_len))]
            music = muspy.Music(resolution=resolution, tracks=[track])
        if mode == 'note':
            # Note-based representation:
            # (time, pitch, duration, velocity) for each note, used as 4 channels in ResNet
            return music.to_note_representation(), resolution
        else:
            return music.to_event_representation(), resolution
    except Exception as e:
        print(f'Failed to read file {file}!')
        print(e)
        return None, None


def events_to_midi(events, resolution=None):
    """
    Convert a numpy array containing a event-based representation sequence to a midi object.
    """
    if resolution is None:
        music = muspy.from_event_representation(events)
    else:
        music = muspy.from_event_representation(events, resolution=resolution)
    return music


def clean(sequence):
    while True:
        i = len(sequence) - 1
        if sequence[i] == 355:
            del (sequence[i])
        else:
            break
    sequence.append(355)

    while True:
        if sequence[0] == 355:
            del (sequence[0])
        else:
            break
    return sequence


def midi_to_txt(path, output_dir):
    music, _ = midi_to_events(path, mode='event')
    text = music.tolist()
    text = reduce(add, text)
    text = clean(text)
    file_name = os.path.split(path)[-1]
    save_name = file_name.replace('midi', 'txt')
    save_path = os.path.join('..', output_dir, save_name)
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
    s = events_to_midi(np.array(data))
    save_name = os.path.split(path)[-1].replace('txt', 'midi')
    out_path = os.path.join('..', output_dir, save_name)
    s.write_midi(out_path)


def get_path(composer):
    base = 'D:\\Code\\Stego\\Project\\maestro-v3.0.0'
    with open(os.path.join(base, 'maestro-v3.0.0.csv'), 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        reader = list(reader)
    del (reader[0])
    files = []
    for line in reader:
        if line[0] == composer:
            files.append(line[4])
    paths = []
    for file in files:
        path = os.path.join(base, file)
        path = path.replace('/', '\\')
        paths.append(path)
    return paths


if __name__ == '__main__':
    composers = ['Franz Schubert', 'Johann Sebastian Bach']
    for composer in composers:
        if os.path.exists(composer):
            print('文件夹{}已存在'.format(composer))
        else:
            os.makedirs(composer)
            print('创建文件夹{}'.format(composer))
        paths = get_path(composer)

        tmp_files = os.listdir(composer)
        done_files = []
        for file in tmp_files:
            file = file.split('\\')[-1].replace('txt', 'midi')
            done_files.append(file)

        i = 1
        for path in paths:
            if path.split('\\')[-1] not in done_files:
                midi_to_txt(path, composer)
            print('{}/{}'.format(i, len(paths)))
            i += 1

    # txt_to_midi(path='versions/tmp/hide/bin.txt', output_dir='versions/midi')
    # midi_to_txt(path='versions/midi/result2.midi', output_dir='versions')
