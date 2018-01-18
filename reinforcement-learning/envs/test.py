#%%
import gym
import pyscreenshot as psc
import pytesseract as pt
import pyautogui as pag
import threading
import time
import matplotlib.pyplot as plt
from collections import deque
from gym import spaces
from gym.utils import seeding

im = psc.grab(bbox=(100,164,207,188))
im.save("/home/string/image.png")
price = pt.image_to_string(im).replace(" ", "")

print(price)
