import argparse
import base64

from Crypto.Cipher import AES


def read_file(file_name):
    with open(file_name, 'rb') as f:
        bytes_stream = f.read()
    return bytes_stream


def split(bytes_stream):
    nonce = base64.decodebytes(bytes_stream[: 17])
    # print(nonce)
    cipher_text = base64.decodebytes(bytes_stream[17: (len(bytes_stream) - 25)])
    tag = base64.decodebytes(bytes_stream[(len(bytes_stream) - 26):])
    return nonce, cipher_text, tag


class Encryption(object):

    def __init__(self, key):
        self.key = key
        self.header = b'header'

    def encrypt(self, byte_data):
        nonce = b'_\xc0\xcf\x9bd\xac!}\xad\xe0\x17\x17'
        # print('The message is: {}'.format(byte_data))

        cipher = AES.new(self.key, AES.MODE_GCM, nonce=nonce)
        cipher.update(self.header)
        cipher_text, tag = cipher.encrypt_and_digest(byte_data)

        nonce = base64.encodebytes(nonce)
        cipher_text = base64.encodebytes(cipher_text)
        # print('The ciphertext is: {}'.format(cipher_text))
        tag = base64.encodebytes(tag)
        return nonce + cipher_text + tag

    def decrypt(self, file_path):
        # print(self.key.decode('utf-8'))
        try:
            bytes_stream = read_file(file_path)
            # print(bytes_stream)
            nonce, cipher_text, tag = split(bytes_stream)
            cipher = AES.new(self.key, AES.MODE_GCM, nonce=nonce)
            cipher.update(self.header)
            plain_text = cipher.decrypt_and_verify(cipher_text, tag)
            return plain_text

        except (ValueError, KeyError) as e:
            print(e)
            return 'Error! The cipher is modified or the key is wrong.'


if __name__ == '__main__':
    parser = argparse.ArgumentParser('aes-gcm')
    parser.add_argument('--mode', type=str, default='encrypt', help='[encrypt, decrypt]')
    parser.add_argument('--file_path', type=str, default='test.txt', help='The file you want to encrypt/decrypt')
    args = parser.parse_args()

    Key = b'\x8a\x02\xc3|\x08\x8a\x81+\xb7g\x9cF\xd2\x08D['
    if args.mode == 'encrypt':
        my_System = Encryption(Key)
        # start = time.time()
        # for i in range(10000):
        my_System.encrypt(args.file_path)
        '''
        end = time.time()
        duration = end - start
        fsize = os.path.getsize(args.file_path)
        print('speed:{:.2f}MB/s'.format(fsize / (1024 * 1024) / duration * 10000))
        '''

    elif args.mode == 'decrypt':
        my_System = Encryption(Key)
        my_System.decrypt('cipher')
