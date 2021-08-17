import cv2

def crop_img(img, hmin=30, hmax=75):
    img = cv2.imread(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    low = np.array([hmin, 100, 50])
    high = np.array([hmax, 255, 255])
    mask = cv2.inRange(hsv, low, high)
    res = cv2.bitwise_and(img, img, mask=mask)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0.0
    max_c = None
    crop = img.copy()
    for c in cnts[0]:
        area = cv2.contourArea(c)
        if area > max_area:
            max_area = area
            max_c = c
    if max_area != 0.0:
        (x, y, w, h) = cv2.boundingRect(max_c)
        crop = img[y:y + h, x:x + w]
    return crop

crop = crop_img("t1.png")
cv2.imwrite("crop.jpg",crop)
