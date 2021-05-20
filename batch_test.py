import random
import string

from predict import *


def gene_datas(file_num=100, file_len_base=200):
    sample_eng = string.ascii_letters + string.digits
    sample_eng += '!@#$%^&*()_+-={}|[]\\:"<>?;\',./'
    punc = '！@#￥%……&*（）——+-={}【】|、：“”；‘’，。、《》？ '
    pos = []
    index_1 = random.randint(0, 5)
    index_2 = random.randint(5, 10)
    index_3 = random.randint(10, 15)
    for i in range(0, 20):
        if i == index_1 or i == index_2 or i == index_3:
            pos.append(True)
        else:
            pos.append(False)
    path = './versions/gen/msg'
    eng = round(file_num / 2)
    for i in range(file_num):
        lens = random.randint(-10, 500)
        lens += file_len_base
        if i < eng:  # 前一半生成英文文本
            chars = random.choices(sample_eng, k=lens)
            cont = ''.join(chars)
        else:  # 后一半生成中文文本
            cont = ''
            for no in range(lens):
                head = random.randint(0xb0, 0xf7)
                body = random.randint(0xa1, 0xfe)
                val = f'{head:x} {body:x}'
                char = bytes.fromhex(val).decode('gbk', 'ignore')
                cont += char.encode('utf8').decode('utf8')
                flag = random.choice(pos)
                if flag:
                    cont += random.choice(punc)
        file = path + '/' + str(i) + '.txt'
        with open(file, 'w', encoding='utf8') as f:
            f.write(cont)
    print('test data has been built.')


def gen_test():
    gene_datas(file_num=100, file_len_base=300)
    # 实际字符长度是file_len_base + rand(-10, 500)
    tests = os.listdir('./versions/gen/msg/')
    for file in tests:
        tar = './versions/gen/msg/' + file
        base = random.randint(2, 6)
        # 窗口值[2, 6)
        hide(
            message=tar,
            window=base,
            seq=[data.note2id[i] for i in data.melodies[6]][: 128],
            output_dir='versions/gen/midi'
        )


if __name__ == '__main__':
    gen_test()
