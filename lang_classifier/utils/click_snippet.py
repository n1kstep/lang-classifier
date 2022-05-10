# pylint: disable=missing-module-docstring

import time

from pynput.mouse import Button, Controller

MouseClick = Controller()

while True:
    MouseClick.click(Button.left, 1)
    print("Clicked")
    time.sleep(60)
