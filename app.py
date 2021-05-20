import os

import muspy
import streamlit as st
import torch
from sklearn.preprocessing import scale

import lzma
from discriminator.convert import midi_to_events
from encrypt import Encryption
from model.predict import hide_app, extract_app, data

TMP_DIR = os.path.join(os.curdir, 'tmp')


def main():
    st.title('MIDI Steganography')
    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Choose the app mode", ["Hide", "Extract", "Detect"])
    if app_mode == "Detect":
        run_detect()
    else:
        run_hide_extract(app_mode)


def play_midi(midi):
    music = muspy.read_midi(midi)
    save_wav = os.path.join(os.path.dirname(midi), 'tmp.wav')
    music.write_audio(path=save_wav, soundfont_path=os.path.join('tmp/MuseScore_General.sf3'),
                      audio_format='wav')
    with open(save_wav, 'rb') as f:
        audio = f.read()
    st.audio(audio, 'audio/wav')


def show_sidebar():
    config = {'cipher': st.sidebar.selectbox(label='Select a cipher', options=['AES-256', 'Chacha20']),
              'key': st.sidebar.text_input(label='Input the key', type='password')}
    return config


def aes_cipher(mode, key, content):
    cipher = Encryption(key)
    if mode == 'Hide':
        return cipher.encrypt(content)
    else:
        return cipher.decrypt(content)


def run_hide_extract(mode):
    config = show_sidebar()

    placeholder = st.empty()
    file = st.file_uploader(label='Upload file')
    placeholder_hide = st.empty()

    selected = st.selectbox(label='Select a beginning', options=list(range(1, len(data.melodies))))

    click = st.button(mode)
    if mode == 'Hide':
        with placeholder.beta_container():
            st.markdown('## Hide something in a piece of music...')
            st.write('Upload your secret file')
        with placeholder_hide.beta_container():
            st.write('Or enter your secret message')
            msg = st.text_input(label='Input').encode('utf-8')
        if click:
            raw = file.getvalue() if file is not None else msg
            compressed = lzma.compress(raw)
            encrypted = aes_cipher(mode, config['key'].encode('utf-8'),
                                   compressed if len(compressed) < len(raw) else raw)
            midi_hide = hide_app(encrypted, int(selected))
            play_midi(midi_hide)
            # TODO: download()
    else:
        with placeholder.beta_container():
            st.markdown('## Extract your message from a MIDI file...')
            st.write('Upload the file with secret message')
        if click:
            with open('tmp/extract/tmp.midi', 'wb') as f:
                f.write(file.getvalue())

            extracted = extract_app('tmp/extract/tmp.midi', int(selected))
            text = aes_cipher(mode, config['key'].encode('utf-8'), extracted)
            try:
                text = lzma.decompress(text)
            except Exception:
                st.warning('Data may not be compressed.')
            with open('tmp/extract/result.out', 'wb') as f:
                f.write(text)
            st.success('Message saved.')
            try:
                st.success(text.decode('utf-8'))
            except Exception:
                pass
            # TODO: download() or show


def run_detect():
    model = torch.jit.load('discriminator/steg.pt')
    labels = ['0', '1']

    st.markdown('## Is this file hacked?')

    st.write('Upload a MIDI file')
    midi = st.file_uploader(label='Upload', type=['mid', 'midi'])

    st.write('Or try an example')
    files = os.listdir(os.path.join(TMP_DIR, 'test'))
    files = [f for f in files if f.split('.')[-1] in ['mid', 'midi']]
    selected = st.selectbox(label='Select an example', options=files)

    click = st.button('Start')

    if midi is not None:
        name = midi.name
        bytes_data = midi.read()
        save_path = os.path.join(TMP_DIR, 'test', name)
        with open(save_path, 'wb') as f:
            f.write(bytes_data)
        # data = read_midi(save_path)
    else:
        name = selected
        save_path = os.path.join(TMP_DIR, 'test', name)

    data, _ = midi_to_events(save_path)

    if click:
        if data is not None:
            input = torch.tensor(scale(data).T).unsqueeze(0).float()
            output = model(input)
            pred = torch.argmax(output, dim=-1)
            st.info(name + ': ' + labels[pred.item()])
            play_midi(save_path)
        else:
            st.error('Something went wrong when reading the uploaded file!')


if __name__ == '__main__':
    main()
