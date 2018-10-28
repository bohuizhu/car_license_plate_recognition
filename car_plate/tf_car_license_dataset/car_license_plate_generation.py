
#coding=utf-8

import os

import pygame

import random

from pygame.locals import *


pygame.init()

# =["SA","VIC","TA"]

number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',

'V', 'W', 'X', 'Y', 'Z']
#list=['k','7','B','6','1']


TRAINSIZE = 10 
FILE_PATH =r".\car_plate"

def save(filename, contents):

    with open(filename,'w') as f:

        f.write(contents)

 
def random_text(char_set=number, captcha_size=4):

    rtext = []

    #rtext.append(random.choice(ALPHABET))

    for i in range(0, captcha_size+1):

     #   c = random.choice(char_set)

        rtext.append(list[i])

    str = ''.join(rtext)

    return str

 

def write_label(instr):

    label_filename = os.path.join(FILE_PATH, "chepai/labels.txt")

    with open(label_filename, "a") as f:

        f.writelines(instr + ' ')

 

font = pygame.font.Font(None, 32) 

filepath = os.path.join(FILE_PATH, "chepai/images")

fileformate = ".jpg"

if not os.path.exists(filepath):

    os.makedirs(filepath)



for x in range(TRAINSIZE):

    text = random_text()

    ftext = font.render(text, True, (65, 83, 130), (255, 255, 255))

    filename = filepath + text + fileformate

    label = text + fileformate + "," + text
    pygame.image.save(ftext, os.path.join(filepath, text + ".jpg"))

    
    write_label(label)

    print(label)

 
