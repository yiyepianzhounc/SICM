# -*- coding: utf-8 -*-

def build_glyph_dict():
    glyph_dict = dict()
    with open('data/glyph/CJK.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[:20950]:
            glyph_dict[line.split()[1]] = line.split()[2]
    return glyph_dict


def txt2stroke(text, glyph_dict):
    stroke_v = [[0]*25]
    for ch in text[5:-5]:
        stroke = [0] * 25
        if ch in glyph_dict.keys():
            for g in glyph_dict[ch]:
                # print(g, end='')
                stroke[ord(g)-97] = 1
        stroke_v.append(stroke)
    stroke_v.append([0]*25)
    return stroke_v
