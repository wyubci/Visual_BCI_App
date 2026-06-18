from psychopy import visual, core
import numpy as np
import time

win = visual.Window(size=(1280, 130), pos=(0, 543), color=(0, 0, 0))
pic_path = 'source/icons/logo.png'

# block = visual.ImageStim(win, pos=(-0.82, 0), size=(0.3, 1.8))
# block.image = pic_path
block = visual.Rect(win, pos=(-0.82, 0), size=(0.3, 1.8), fillColor='white')
frq = 1
now_time = 0
timer = core.Clock()
timer.reset()
t0 = time.time()
block.autoDraw = True
while now_time < 10:
    now_time = timer.getTime()
    t1 = time.time()
    # block.opacity = abs(np.sin(2 * np.pi * now_time* frq/2))
    block.opacity = np.sin(2 * np.pi * now_time * frq)
    # opacity 的值 为0~1 大于1取1，小于0取0
    # block.contrast = np.sin(2 * np.pi * now_time * frq)
    # block.draw()
    win.flip()
    print(now_time, t1-t0)
win.close()