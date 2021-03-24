#!/bin/bash

x11vnc -display :0 -rfbauth ~/.vnc/passwd -rfbport 5900 -loop -shared
