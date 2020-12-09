import os
import json
import jieba


def data_process(file_dir, data_type):
    file = os.path.join(file_dir, "%s.json" % data_type)
    zh_file = os.path.join(file_dir, "%s.zh" % data_type)
    en_file = os.path.join(file_dir, "%s.en" % data_type)
    with open(file, 'r', encoding='utf-8') as f, open(zh_file, 'w', encoding='utf-8') as zf, open(en_file, "w", encoding='utf-8') as ef:
        data = json.load(f)
        for en_zh_pair in data:
            ef.write(en_zh_pair[0] + '\n')
            zh_words = list(jieba.cut(en_zh_pair[1]))
            zf.write(' '.join(zh_words) + '\n')


if __name__ == '__main__':
    for data_type in ['train', 'dev', 'test']:
        file_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')
        data_process(file_dir, data_type)
