import numpy as np
from PIL import Image

from PyV4L2Camera.camera import Camera
from PyV4L2Camera.controls import ControlIDs

camera = Camera('/dev/video0', 1920, 1080)
controls = camera.get_controls()

for control in controls:
    print(control.name)

camera.set_control_value(ControlIDs.BRIGHTNESS, 48)

for _ in range(2):
    frame = camera.get_frame()

    # Decode the image
    im = Image.frombytes('RGB', (camera.width, camera.height), frame, 'raw',
                         'RGB')

    # Convert the image to a numpy array and back to the pillow image
    arr = np.asarray(im)
    im = Image.fromarray(np.uint8(arr))

    print(np.mean(arr[:,:,0]))
    print(np.mean(arr[:,:,1]))
    print(np.mean(arr[:,:,2]))

    # Display the image to show that everything works fine
    im.show()

camera.close()
