import time
from pynput.mouse import Controller ,Button

MouseClick = Controller()

while True:
    MouseClick.click(Button.left, 1)
    print('Clicked')
    time.sleep(60)
