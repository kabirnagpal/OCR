import cv2
import numpy as np
import pickle
import imutils


img = cv2.imread("/home/jak/Desktop/myhandwriting/final/image12.jpg",0)
cv2.imshow("img", imutils.resize(img, height = 650))
cv2.waitKey(0)
cv2.destroyAllWindows()
gray=img


_,thresh = cv2.threshold(gray,150,255,cv2.THRESH_BINARY_INV) 
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
dilated = cv2.dilate(thresh,kernel,iterations = 2)
contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) 



def resize2SquareKeepingAspectRation(img, interpolation, size=28):
    h, w = img.shape[:2]
    c = None if len(img.shape) < 3 else img.shape[2]
    if h == w: return cv2.resize(img, (size, size), interpolation)
    if h > w: dif = h
    else:     dif = w
    x_pos = int((dif - w)/2.)
    y_pos = int((dif - h)/2.)
    if c is None:
        mask = np.zeros((dif, dif), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
    else:
        mask = np.zeros((dif, dif, c), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]
    return cv2.resize(mask, (size, size), interpolation)



def tell_the_word(image,model):
    image = resize2SquareKeepingAspectRation(image, cv2.INTER_AREA)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(1,1))
    image = cv2.dilate(image,kernel,iterations = 1)
    #cv2.imshow("img", image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    array=[]
    array.insert(0,image)
    arr=np.asarray(array)
    arr = arr.reshape(1,28,28,1)
    p=model.predict(arr)
    mylist=np.array(p).tolist()
    ind = np.argmax(mylist)
    return ind


with open('/home/jak/Desktop/myhandwriting/models/writing99', 'rb') as f:
    model = pickle.load(f)

for contour in contours:
    [x,y,w,h] = cv2.boundingRect(contour)
    
    if h>150 or w>150:
        continue
    if h<30 or w<30:
        continue
    
    if x>=img.shape[0] or y>=img.shape[1]:
        continue
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1)
    x=x-5
    y=y-5
    w=w+10
    h=h+10
    lol = dilated[y:y+h,x:x+w]
    lol.astype('uint8')
    alpha = tell_the_word(lol,model)
    cv2.putText(img, chr(65 + alpha), (x+5,y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1)
    print(chr(65+alpha))
    
    
cv2.imshow("Scanned", imutils.resize(img, height = 650))
cv2.waitKey(0)
cv2.destroyAllWindows()