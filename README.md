# AIRIS Minecraft

# Requires Python 3.10

## Installation Instructions

1. Clone this repo
2. Install Minecraft
3. Install CurseForge
4. Open CurseForge and search for Vereya
5. Select Vereya and install it under a new profile
6. (recommended) click the profile (not Play), click the 3 dots to the left of Play, click "open folder", open "options.txt", find pauseOnLostFocus and change the value to "false". This prevents Minecraft from pausing when clicking on other windows.
7. Select the profile in CurseForge and click Play
8. In the Minecraft Launcher, click Play
9. Once Minecraft has loaded to the main menu, run main.py
10. (optional) Run MinecraftEye.py

# MinecraftEye

Voxel Engine (like Minecraft) in Python and OpenGL that takes data from AIRIS and shows where it has explored and what it is thinking.

Run MinecraftEye.py after starting Minecraft. Once the agent starts moving, the world will automatically update. Press F to instantly move to where the agent currently is. Press F again to regain camera control.

# Controls: 

Move: W A S D

Fly up: E

Fly down: Q

Fly fast: Hold shift

Toggle Auto-Camera: F

Switch View Filter: M

Toggle X-Ray mode: X

Toggle "View All States" (Can get very laggy after a lot of exploring): B

Release cursor: Esc

Capture cursor: Enter

# View Description

Small Blue Cube: The agents current location

Large Blue Cube: The blocks that the agent can currently see

Red Arrows: Where the agent predicts it will go when performing a particular action, but is not confident

Black Arrows: Where the agent predicts it will go when performing a particular action, and is confident

Green Arrows: The predicted outcome of the action it is going to perform. Can be a chain that represents a sequence of planned actions.

Small White Cube: Locations the agent has been that it searched to create the current plan

Small Green Cube: Locations the agent is expecting to go as part of its plan

Large Green Cube: The area of blocks the agent is expecting to discover at the end of its plan

Small Black Cube (Only visible if "View All States" is on): Every location the agent has been


All AIRIS code is contained in airis.py
