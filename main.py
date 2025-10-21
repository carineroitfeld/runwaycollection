import cv2
import os
import time

myPath = r'C:\Users\sahil\Downloads\New_folder'  # Use raw string to avoid escape character issues
cameraNo = 0  # Not used anymore with IP camera
cameraBrightness = 190
moduleVal = 10
minBlur = 20
grayImage = False
saveData = True
showImage = True
imgWidth = 640
imgHeight = 420
global countFolder

# Replace the IP address with your actual IP camera's URL
#ip_camera_url = "http://192.168.1.6:8080/video"  # or "rtsp://<ip_address>:<port>/stream"
cap = cv2.VideoCapture(0)  # Use the variable, not the string
cap.set(3, imgWidth)
cap.set(4, imgHeight)
cap.set(10, cameraBrightness)
count = 0
countSave = 0


def saveDataFunc():
    global countFolder
    countFolder = 0
    while os.path.exists(myPath + str(countFolder)):
        countFolder += 1
    os.makedirs(myPath + str(countFolder))


if saveData:
    saveDataFunc()

while True:
    success, img = cap.read()
    if not success:
        print("Error: Could not read frame.")
        break

    img = cv2.resize(img, (imgWidth, imgHeight))

    if grayImage:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if saveData:
        if grayImage:
            blur = cv2.Laplacian(img, cv2.CV_64F).var()
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.Laplacian(gray, cv2.CV_64F).var()

        if count % moduleVal == 0 and blur > minBlur:
            nowTime = time.time()
            cv2.imwrite(
                myPath + str(countFolder) + '/' + str(countSave) + " " + str(int(blur)) + " " + str(nowTime) + ".png",
                img)
            countSave += 1

    count += 1

    if showImage:
        cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()