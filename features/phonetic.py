# -*- coding: utf-8 -*-

from pypinyin import lazy_pinyin, Style
from pypinyin.contrib.tone_convert import to_initials, to_finals

def build_pinyin_map():
    aux_map = {'b': [[1, 0], 0], 'p': [[1, 0], 1], 'm': [[2, 0], 0], 'f': [[3, 0], 0],
               'd': [[4, 0], 0], 't': [[4, 0], 1], 'n': [[5, 0], 0], 'l': [[6, 0], 0],
               'g': [[7, 0], 0], 'k': [[7, 0], 1], 'h': [[8, 0], 0], 'j': [[4, 9], 0],
               'q': [[4, 9], 1], 'x': [[9, 0], 0], 'zh': [[4, 11], 0], 'ch': [[4, 11], 1],
               'sh': [[11, 0], 0], 'r': [[10, 0], 0], 'z': [[4, 11], 0], 'c': [[4, 11], 1],
               's': [[11, 0], 0]
               }
    vow_map = ['a', 'o', 'e', 'i', 'u', 'v']
    return aux_map, vow_map


def txt2pinyin(text, aux_map, vow_map):
    try:
        pinyin = lazy_pinyin(text[5:-5], style=Style.TONE3, errors=lambda x: len(x))
    except TypeError:
        print(text)
    pinyin_v = [[0]*25]

    for py in pinyin:
        aux_v = [0] * 13
        vow_v = [0] * 8
        tune_v = [0] * 4
        if type(py) is not int:
            if py[-1].isdigit():
                tune_v[int(py[-1]) - 1] = 1

            aux = to_initials(py)
            if aux:
                if aux in ['zh, sh, ch']:
                    aux_v[11] = 1
                aux_v[12] = aux_map[aux][1]
                aux_v[aux_map[aux][0][0] - 1] = 1
                if aux_map[aux][0][1] != 0:
                    aux_v[aux_map[aux][0][1] - 1] = 1

            vow = to_finals(py)
            if len(vow) >= 3 and vow[-3:] in ['ang', 'eng', 'ong', 'ing']:
                vow = vow[:-2]
                vow_v[7] = 1
            elif len(vow) >= 2 and vow[-2:] in ['an', 'en', 'vn', 'in', 'un']:
                vow = vow[:-1]
                vow_v[6] = 1
            elif vow == 'er':
                vow = 'e'
            for v in vow:
                try:
                    vow_v[vow_map.index(v)] = 1
                except ValueError:
                    print(v, text)

            vow_v.extend(tune_v)
            aux_v.extend(vow_v)
            pinyin_v.append(aux_v)
        else:
            for i in range(py):
                pinyin_v.append([0]*25)
    pinyin_v.append([0]*25)

    return pinyin_v