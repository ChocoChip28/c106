import cv2


# Create our body classiier
classifier = cv2.CascadeClassifier("haarcascade_fullbody.xml")

# Initiate video capture for video file
cap = cv2.VideoCapture('walking.avi')

# Loop once video is successfully loaded
while True:
    
    # Read first frame
    ret, frame = cap.read()

    #Convert Each Frame into Grayscale
    grey = cv2.cvtColor( frame , cv2.COLOR_BGR2GRAY)
    # Pass frame to our body classifier
    body = classifier.detectMultiScale(grey)
    for (x,y,width,height) in body:
        cv2.rectangle(frame,(x,y),(x+width,y+height),(150,200,220),2)
        
    cv2.imshow("body",frame)

    
    # Extract bounding boxes for any bodies identified
    

    if cv2.waitKey(1) == 32: #32 is the Space Key
        break

cap.release()
cv2.destroyAllWindows()
