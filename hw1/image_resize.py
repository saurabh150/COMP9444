#!/usr/bin/python
from PIL import Image
import os, sys

path = "./test/plot/"
dirs = os.listdir( path )

def resize():
    for item in dirs:
        if (item.split('.')[-1] == 'png'):
            if os.path.isfile(path+item):
                im = Image.open(path+item)
                f, e = os.path.splitext(path+item)
                im = im.resize((75,75), Image.ANTIALIAS)
                im.save(path+item)

resize()