import cv2
import numpy as np
from keras.models import load_model

Total = 15
Totalim = 7

thr = 0.999

#function predict takes an array of probabilities, 
#compare the max proba to the threshold value (0.9 in this case)to return result (intruder or user)
def predict(predictions):
    if np.max(predictions, axis=1) < thr:
        
        return "intruder"
    else:
        return np.argmax(predictions, axis=1)

#reads image, preprocess it then pass it to the model for prediction
def readpred(path):
    
    image = cv2.imread(path)
    image = cv2.resize(image, (154, 102))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = image / 255.0
    image = np.array(image)
    image = image.reshape(1, 102, 154, 1)

    predictions = model.predict(image)
    
    return predict(predictions)


model = load_model("model_both.h5")
model.summary()

#loop through images of six impostors right/left qnd pass them to readpred func


def frr():

    RightFRR = 0
    i = 0
    while i <= 14:

        pathr = "test/eyes"+str(i)+".jpg"

        
        pred = readpred(pathr)

        pred = str(pred)


        #print(i-1," is ",pred)
        #print(i-1," is ",predl)
        pred = pred.replace("[","")
        pred = pred.replace("]","")


        if pred =="intruder":
            RightFRR +=1
    
        i+=1
    FRRr = RightFRR/Total
    print("right FRR ",FRRr)

def far():
    LeftFaR = 0
    RightFaR = 0
    i = 15
    while i <= 21:

        pathr = "testim/eyes"+str(i)+".jpg"


        pred = readpred(pathr)
  
        pred = str(pred)
        pred = pred.replace("[","")
        pred = pred.replace("]","")
        

        if pred !="intruder":
            RightFaR +=1

    
    
        i+=1
    FaRr = RightFaR/Totalim


    print("right FAR ",FaRr)


frr()
far()

