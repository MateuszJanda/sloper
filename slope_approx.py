import numpy as np
import cv2

# Create a black image
img = np.zeros((512,512,3), np.uint8)

# Draw a diagonal blue line with thickness of 5 px
cv2.line(img,(0,0),(511,511),(255,0,0),5)

# cv2.putText(img, "Hello world!", )
x = 0
y = 100
cv2.putText(img,"Hello World!!!", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()