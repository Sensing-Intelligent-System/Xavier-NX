#!/bin/bash

Xvfb :1 -screen 0 1600x1200x16 &
export DISPLAY=:1

x11vnc -display :1 -rfbauth ~/.vnc/passwd -rfbport 5900 -loop -shared
