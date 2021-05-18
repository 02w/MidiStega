import os
import csv
import muspy
import numpy as np

MELODY_NOTE_OFF = 0


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
        path = 'midi_data' + '/' + file
        paths.append(path)
    return paths


# 补充缺失值，修正类型
def modify(music):
    for track in music.tracks:
        for note in track.notes:
            note.pitch = int(note.pitch.tolist()[0])
    '''
    for tempo in music.tempos:
        tempo.time = 0
        tempo.qpm = 120
    
    for time_signature in music.time_signatures:
        time_signature.time = 0
        time_signature.numerator = 4
        time_signature.denominator = 4
    '''
    return music


def clean(sequence):
    while True:
        i = len(sequence) - 1
        if sequence[i] == 0:
            del (sequence[i])
        else:
            break
    sequence.append(0)

    while True:
        if sequence[0] == 0:
            del (sequence[0])
        else:
            break
    return sequence


def standardizing(path, out_dir):
    music = muspy.read_midi(path)
    music.adjust_resolution(factor=0.03125)
    sequence = music.to_pitch_representation()
    old_music = muspy.from_pitch_representation(sequence, resolution=15)
    modify(old_music)
    save_name = path.split('/')[-1]
    save_path = out_dir + '/' + save_name
    muspy.write_midi(save_path, old_music)


def midi2txt(path, out_dir):
    music = muspy.read_midi(path)
    sequence = music.to_pitch_representation()
    sequence = sequence.reshape(len(sequence))
    sequence = sequence.tolist()
    clean(sequence)
    save_name = path.split('/')[-1].replace('midi', 'txt')
    save_path = out_dir + '/' + save_name

    with open(save_path, 'w') as f:
        for i in range(len(sequence)):
            f.write(str(sequence[i]))
            if i != len(sequence) - 1:
                f.write(' ')


def txt2midi(path, out_dir):
    with open(path, 'r') as f:
        text = f.read().strip().split(' ')
        sequence = []
        for item in text:
            sequence.append(int(item))

    sequence = np.array(sequence)
    sequence.reshape(len(sequence), 1)
    music = muspy.from_pitch_representation(sequence, resolution=15)
    # modify(music)
    save_name = path.split('/')[-1].replace('txt', 'midi')
    save_path = out_dir + '/' + save_name
    muspy.write_midi(save_path, music)


if __name__ == '__main__':
    composers = ['Franz Schubert', 'Johann Sebastian Bach', 'Wolfgang Amadeus Mozart', 'Ludwig van Beethoven']
