from .convert import *
from random import randint

# 学习指定音乐家midi文件的
def learn():
    for composer in ['Franz Schubert', 'Johann Sebastian Bach']:
        paths = get_path(composer)
        statistic = {}
        for path in paths:
            music = muspy.read_midi(path)
            music.adjust_resolution(factor=0.03125)
            for note in music.tracks[0].notes:
                if note.pitch not in statistic.keys():
                    statistic[note.pitch] = {}
                statistic[note.pitch][note.velocity] = statistic[note.pitch].get(note.velocity, 0) + 1
        result = {}
        save_path = composer + '_habit.pkl'
        joblib.dump(statistic, save_path)

def streamline(statistic):
    result = {}
    for pitch, child_dict in statistic.items():
        ranked_dict = sorted(child_dict.items(), key=lambda x: x[1], reverse=True)
        result[pitch] = []
        cnt = 0
        for pair in ranked_dict:
            result[pitch].append(pair[0])
            cnt += 1
            if cnt >= 5:
                break
    return result

def get_velocity(pitch, habit, timer):
    if pitch not in habit.keys():
        return 64, timer

    else:
        candidates = habit[pitch]
        length = len(candidates)
        for candidate in candidates:
            if timer[pitch][candidate] != 0:
                winner = candidate
                timer[pitch][winner] -= 1
                return winner, timer

        index = randint(0, length - 1)  # 更新velocity
        winner = candidates[index]
        timer[pitch][winner] = 5
        return winner, timer

def iseven(number):
    if number % 2 == 0:
        return True
    else:
        return False

def modify(music):
    timer = {}
    for i in range(0, 130):
        timer[i] = {}
        for j in range(0, 130):
            timer[i][j] = 0

    habit = streamline(joblib.load('model/habit.pkl'))
    sequence = music.to_pitch_representation()
    length = len(sequence)
    print(length)
    sequence = sequence.reshape(length)
    sequence = sequence.tolist()
    new_sequence = []

    for pitch in sequence:
        for i in range(6):
            new_sequence.append(pitch)

    new_sequence = np.array(new_sequence)
    new_sequence.reshape(len(new_sequence), 1)
    new_music = muspy.from_pitch_representation(new_sequence, resolution=15)

    for note in new_music.tracks[0].notes:
        note.velocity, timer = get_velocity(note.pitch, habit, timer)

        if note.velocity > 80:
            note.velocity -= 20

        if note.pitch > 60:
            note.pitch -= 20
            if not iseven(note.velocity):
                note.velocity += 1

        else:
            if iseven(note.velocity):
                note.velocity += 1
    return new_music

def de_modify(music):
    for note in music.tracks[0].notes:
        if iseven(note.velocity):
            note.pitch += 20

    sequence = music.to_pitch_representation()
    sequence = sequence.reshape(len(sequence))
    sequence = sequence.tolist()
    new_sequence = []

    length = int(len(sequence) / 6)
    for i in range(length):
        j = i * 6
        new_sequence.append(sequence[j])
    new_sequence = np.array(new_sequence)
    new_sequence = new_sequence.reshape(len(new_sequence), 1)
    new_music = muspy.from_pitch_representation(new_sequence, resolution=15)
    return new_music


if __name__ == '__main__':
    music = muspy.read_midi('0.midi')

