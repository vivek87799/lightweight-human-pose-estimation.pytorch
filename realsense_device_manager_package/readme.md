# Box dimensions calculation using multiple realsense camera

## Requirements: 
### Python Version
This code requires Python 3.6 to work and does not work with Python 2.7.

### Packages: 
1. OpenCV
2. LibRealSense
3. Numpy

Get these packages using pip: pip install opencv-python numpy pyrealsense2


## Aim
This module uses the SDK for aligning multiple devices to a unified co-ordinate system in world. 


## Workflow
1. Place the calibration chessboard object into the field of view of all the realsense cameras. Update the chessboard parameters in the script in case a different size is chosen.                                 
2. Start the program.

## Example Output
Once the calibration is done and the target object's dimensions are calculated, the application will open as many windows as the number of devices connected each displaying a color image along with an overlay of the calculated bounding box.
In the following example we've used two Intel® RealSense™ Depth Cameras D435 pointing at a common object placed on a 6 x 9 chessboard (checked-in with this demo folder).
![sampleSetupAndOutput](https://github.com/framosgmbh/librealsense/blob/box_dimensioner_multicam/wrappers/python/examples/box_dimensioner_multicam/samplesetupandoutput.jpg)

## References
Rotation between two co-ordinates using Kabsch Algorithm: 
Kabsch W., 1976, A solution for the best rotation to relate two sets of vectors, Acta Crystallographica, A32:922-923

