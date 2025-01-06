# Infinity-Blade-Image-Processing-Project
Real-time video-based program interfacing with the computer game Infinity Blade. The player uses hand and body gestures to play the game.

main: Main code that interfaces with the game and sends keys according to the player's actions:

dodge: The player can dodge to the right or to the left by moving.
block: The player blocks by holding the sword horizontally over their head for a few seconds.
attack: By moving the sword in various directions, the player attacks. All 8 directions are being detected - up, down, right, left, up-right, down-left, etc.
color_detector: Code for detecting the sword's color in the HSV colorspace by applying maximum and minimum thresholds for each channel.

color_picker: Code for selecting a color mask for the color of the sword, using the thresholds detected in the color_detector file.

kalmanfilter: Code implementing the Kalman filter for following the sword and predicting its next location.
