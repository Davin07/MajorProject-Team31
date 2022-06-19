import numpy as np
from keras.models import model_from_json
import operator
import cv2
import sys, os
from gtts import gTTS  
from time import time, sleep

def main():

    # Loading the model
    #json_file = open("Models/CNN/cnn_model.json", "r")
    json_file = open("models/trsl/CNNThresholdTrSL.json", "r")
    model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(model_json)
    # load weights into new model
    #loaded_model.load_weights("Models/CNN/CNN.h5")
    loaded_model.load_weights("models/trsl/CNNThresholdTrSL.h5")
    print("Loaded model from disk")

    cap = cv2.VideoCapture(0)

    # Category dictionary
    # categories = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    #                 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V',
    #                 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'del', 27: 'nothing', 28: 'space'}
    categories = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
                    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'R', 17: 'S', 18: 'T', 19: 'U', 20: 'V', 21: 'Y',
                    22: 'Z', 23: 'DELETE', 24: 'NOTHING', 25: 'SPACE'}      #trsl
    # categories = {0: '1', 1: '2', 2: '3', 3: '4', 4: '5', 5: '6', 6: '7', 7: '8', 8: '9', 9: 'A',
    #                 10: 'B', 11: 'C', 12: 'D', 13: 'E', 14: 'F', 15: 'G', 16: 'H', 17: 'I', 18: 'J', 19: 'K', 20: 'L', 21: 'M',
    #                 22: 'N', 23: 'O', 24: 'P', 25: 'Q', 26: 'R', 27: 'S', 28: 'T', 29: 'U', 30:'V', 31:'W', 32:'X', 33:'Y', 34:'Z'}        #isl
    # categories = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    #                 10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J', 20: 'K', 21: 'L',
    #                 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T', 30:'U', 31:'V', 32:'W', 33:'X', 34:'Y', 35:'Z'}

    while True:
        _, frame = cap.read()
        # Simulating mirror image
        frame = cv2.flip(frame, 1)
        
        # Coordinates of the ROI
        x1 = int(0.5*frame.shape[1])
        y1 = 10
        x2 = frame.shape[1]-10
        y2 = int(0.5*frame.shape[1])
        # Drawing the ROI
        # The increment/decrement by 1 is to compensate for the bounding box
        cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
        # Extracting the ROI
        roi = frame[y1:y2, x1:x2]
        
        # Resizing the ROI so it can be fed to the model for prediction
        roi = cv2.resize(roi, (64, 64)) 
        roi = cv2.flip(roi, 1)

        #_, test_image = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
        #test_image = cv2.resize(test_image, (300,300))
        #cv2.imshow("test", test_image)

        cv2.imshow("ROI", roi)
        
        minValue = 70
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(5,5),2)
        th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
        ret, test_image = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        test_image = cv2.resize(test_image, (64,64))
        cv2.imshow("test", test_image)


        result = loaded_model.predict(test_image.reshape(1, 64, 64, 1))
        # prediction = {'A': result[0][0], 'B': result[0][1], 
        #             'C': result[0][2], 'D': result[0][3],
        #             'E': result[0][4], 'F': result[0][5],
        #             'G': result[0][6], 'H': result[0][7],
        #             'I': result[0][8], 'J': result[0][9],
        #             'K': result[0][10], 'L': result[0][11],
        #             'M': result[0][12], 'N': result[0][13],
        #             'O': result[0][14], 'P': result[0][15],
        #             'Q': result[0][16], 'R': result[0][17],
        #             'S': result[0][18], 'T': result[0][19],
        #             'U': result[0][20], 'V': result[0][21],
        #             'W': result[0][22], 'X': result[0][23],
        #             'Y': result[0][24], 'Z': result[0][25],
        #             'del': result[0][26], 'nothing': result[0][27],
        #             'space': result[0][28]}
        prediction = {'A': result[0][0], 'B': result[0][1], 
                    'C': result[0][2], 'D': result[0][3],
                    'E': result[0][4], 'F': result[0][5],
                    'G': result[0][6], 'H': result[0][7],
                    'I': result[0][8], 'J': result[0][9],
                    'K': result[0][10], 'L': result[0][11],             
                    'M': result[0][12], 'N': result[0][13],
                    'O': result[0][14], 'P': result[0][15],
                    'R': result[0][16], 'S': result[0][17],
                    'T': result[0][18], 'U': result[0][19],
                    'V': result[0][20], 'Y': result[0][21],
                    'Z': result[0][22], 'DELETE': result[0][23],
                    'NOTHING': result[0][24], 'SPACE': result[0][25]};           #trsl
        # prediction = {'1': result[0][0], '2': result[0][1], 
        #             '3': result[0][2], '4': result[0][3],
        #             '5': result[0][4], '6': result[0][5],
        #             '7': result[0][6], '8': result[0][7],
        #             '9': result[0][8], 'A': result[0][9],
        #             'B': result[0][10], 'C': result[0][11],
        #             'D': result[0][12], 'E': result[0][13],
        #             'F': result[0][14], 'G': result[0][15],
        #             'H': result[0][16], 'I': result[0][17],
        #             'J': result[0][18], 'K': result[0][19],
        #             'L': result[0][20], 'M': result[0][21],             
        #             'N': result[0][22], 'O': result[0][23],
        #             'P': result[0][24], 'Q': result[0][25],
        #             'R': result[0][26], 'S': result[0][27],
        #             'T': result[0][28], 'U': result[0][29],
        #             'V': result[0][30], 'W': result[0][31],
        #             'X': result[0][32], 'Y': result[0][33],
        #             'Z': result[0][34]}                                    #isl
        # prediction = {'0': result[0][0], '1': result[0][1], 
        #             '2': result[0][2], '3': result[0][3],
        #             '4': result[0][4], '5': result[0][5],
        #             '6': result[0][6], '7': result[0][7],
        #             '8': result[0][8], '9': result[0][9],
        #             'A': result[0][10], 'B': result[0][11],
        #             'C': result[0][12], 'D': result[0][13],         //auslan
        #             'E': result[0][14], 'F': result[0][15],
        #             'G': result[0][16], 'H': result[0][17],
        #             'I': result[0][18], 'J': result[0][19],
        #             'K': result[0][20], 'L': result[0][21],
        #             'M': result[0][22], 'N': result[0][23],
        #             'O': result[0][24], 'P': result[0][25],
        #             'Q': result[0][26], 'R': result[0][27],
        #             'S': result[0][28], 'T': result[0][29],
        #             'U': result[0][30], 'V': result[0][31],
        #             'W': result[0][32], 'X': result[0][33],
        #             'Y': result[0][34], 'Z': result[0][35]}
        # print(prediction)
        # Sorting based on top prediction
        prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        
        # Displaying the predictions
        cv2.putText(frame, prediction[0][0], (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)    
        cv2.imshow("Frame", frame)
        #speak_letter(prediction[0][0])
        
        interrupt = cv2.waitKey(10)
        if interrupt & 0xFF == 27: # esc key
            break
            
    
    cap.release()
    cv2.destroyAllWindows()

def speak_letter(letter):
	# Create the text to be spoken
    prediction_text = letter
    
    # Create a speech object from text to be spoken
    speech_object = gTTS(text=prediction_text, lang='en', slow=False)

    # Save the speech object in a file called 'prediction.mp3'
    speech_object.save("prediction.mp3")
 
    # Playing the speech using mpg321
    os.system("start prediction.mp3")


if __name__ == '__main__':
    main()

