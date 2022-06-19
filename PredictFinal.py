#Lastest
import tkinter as tk
from PIL import ImageTk, Image
import cv2
import os
from playsound import playsound
# import pygame
import numpy as np
from keras.models import model_from_json
import operator
import time
import sys, os
from string import ascii_uppercase
from gtts import *
from spellchecker import SpellChecker
# from textblob import TextBlob

class Vid:
    # function for video streaming
    def __init__(self):
        # Loading the CNN model
        self.json_file = open("Models\\ASL\CNN\CNNThreshold.json", "r")
        self.model_json = self.json_file.read()
        self.json_file.close()
        self.loaded_model = model_from_json(self.model_json)
        # load weights into new model
        self.loaded_model.load_weights("Models\ASL\CNN\CNNThreshold.h5")
        print("Loaded ASL model from disk")
        self.arcs = 'asl'
        # self.prediction[0][0] = ''

        
        # initialize a variable as timer
        self.i = 0

        # Initialize spellchecker
        self.spell = SpellChecker()

        # Create Tkinter window
        self.root = tk.Tk()
        self.root.title("Sign language to Text Converter")
        self.root.geometry("1200x1200")

        # Displays "Choose Architecture"
        self.text1 = tk.Label(self.root)
        self.text1.grid(row=0, column=0, columnspan=10)
        self.text1.config(text="Choose Architecture :", font=("Courier",18))
        
        # Create a unit to display all buttons
        self.btns = tk.Frame(self.root, bg="white")
        self.btns.grid(row=1, column=0, columnspan=10)
        
        # Buttons to load each model (calls a function to perform the action)
        self.arch1 = tk.Button(self.btns, text ="ASL", command = self.asl).grid(row=1, column=0)
        self.arch2 = tk.Button(self.btns, text ="ISL", command = self.isl).grid(row=1, column=1)
        self.arch3 = tk.Button(self.btns, text ="TRSL", command = self.trsl).grid(row=1, column=2)
        # self.arch4 = tk.Button(self.btns, text ="Inception", command = self.cnn).grid(row=1, column=3)
        # self.arch5 = tk.Button(self.btns, text ="CNN", command = self.cnn).grid(row=1, column=4)

        # Displays Which model has been loaded
        self.text2 = tk.Label(self.root)
        self.text2.grid(row=2, column=0, columnspan=10)
        self.text2.config(text="Loaded CNN model from disk", font=("Courier",18))

        # Create a labels to show the streams
        # Test stream
        self.imgTest = tk.Label(self.root)
        self.imgTest.grid(row=4, column=0, rowspan=2, columnspan=2)
        # Full stream
        self.imgFrame = tk.Label(self.root)
        self.imgFrame.grid(row=4, column=2, rowspan=6, columnspan=6)

        # Displays the predicted character
        self.text3 = tk.Label(self.root)
        self.text3.grid(row=20, column=0, columnspan=2)
        self.text3.config(text="Character :", font=("Courier",28,"bold"))
        self.textch = tk.Label(self.root) # Current Symbol
        self.textch.grid(row=20, column=2)

        # Displays the word
        self.text4 = tk.Label(self.root)
        self.text4.grid(row=24, column=0, columnspan=2)
        self.text4.config(text="Word :", font=("Courier",28,"bold"))
        self.textWord = tk.Label(self.root) # Current Symbol
        self.textWord.grid(row=24, column=2, columnspan=50)

        # Displays the whole sentence
        self.text5 = tk.Label(self.root)
        self.text5.grid(row=26, column=0, columnspan=2)
        self.text5.config(text="Sentence :", font=("Courier",28,"bold"))
        self.textSentence = tk.Label(self.root) # Current Symbol
        self.textSentence.grid(row=26, column=2, rowspan=10, columnspan=50)

        self.clearText = tk.Button(self.root, text ="Clear Text", command = self.clear).grid(row=28, column=0)

        # initialize word and sentence
        self.word = ""
        self.sentence = ""

        # Capture from camera
        self.cap = cv2.VideoCapture(0)

        # Call the function to show the stream
        self.video_stream()
    
    def clear(self):
        self.word = ''
        self.sentence  =''

    # Functions to Load different models
    def asl(self):
        self.arcs = 'asl'
        # Loading the model
        self.json_file = open("Models\ASL\CNN\CNNThreshold.json", "r")
        self.model_json = self.json_file.read()
        self.json_file.close()
        self.loaded_model = model_from_json(self.model_json)
        # load weights into new model
        self.loaded_model.load_weights("Models\ASL\CNN\CNNThreshold.h5")
        self.result = self.loaded_model.predict(self.test_image.reshape(1, 64, 64, 1))
        self.prediction = {'A': self.result[0][0], 'B': self.result[0][1], 
                    'C': self.result[0][2], 'D': self.result[0][3],
                    'E': self.result[0][4], 'F': self.result[0][5],
                    'G': self.result[0][6], 'H': self.result[0][7],
                    'I': self.result[0][8], 'J': self.result[0][9],
                    'K': self.result[0][10], 'L': self.result[0][11],
                    'M': self.result[0][12], 'N': self.result[0][13],
                    'O': self.result[0][14], 'P': self.result[0][15],
                    'Q': self.result[0][16], 'R': self.result[0][17],
                    'S': self.result[0][18], 'T': self.result[0][19],
                    'U': self.result[0][20], 'V': self.result[0][21],
                    'W': self.result[0][22], 'X': self.result[0][23],
                    'Y': self.result[0][24], 'Z': self.result[0][25],
                    'del': self.result[0][26], 'nothing': self.result[0][27],
                    'space': self.result[0][28]}
        self.prediction = sorted(self.prediction.items(), key=operator.itemgetter(1), reverse=True)
        self.text2.config(text="Loaded ASL model from disk")

    def isl(self):
        self.arcs = 'isl'
        # Loading the model
        self.json_file = open("Models\ISL\CNN\CNNThresholdISL.json", "r")
        self.model_json = self.json_file.read()
        self.json_file.close()
        self.loaded_model = model_from_json(self.model_json)
        # load weights into new model
        self.loaded_model.load_weights("Models\ISL\CNN\CNNThresholdISL.h5")
        self.result = self.loaded_model.predict(self.test_image.reshape(1, 64, 64, 1))
        self.prediction = {'1': self.result[0][0], '2': self.result[0][1], 
                    '3': self.result[0][2], '4': self.result[0][3],
                    '5': self.result[0][4], '6': self.result[0][5],
                    '7': self.result[0][6], '8': self.result[0][7],
                    '9': self.result[0][8], 'A': self.result[0][9],
                    'B': self.result[0][10], 'C': self.result[0][11],
                    'D': self.result[0][12], 'E': self.result[0][13],
                    'F': self.result[0][14], 'G': self.result[0][15],
                    'H': self.result[0][16], 'I': self.result[0][17],
                    'J': self.result[0][18], 'K': self.result[0][19],
                    'L': self.result[0][20], 'M': self.result[0][21],
                    'N': self.result[0][22], 'O': self.result[0][23],
                    'P': self.result[0][24], 'Q': self.result[0][25],
                    'R': self.result[0][26], 'S': self.result[0][27],
                    'T': self.result[0][28], 'U': self.result[0][29],
                    'V': self.result[0][30], 'W': self.result[0][31],
                    'X': self.result[0][32], 'Y': self.result[0][33],
                    'Z': self.result[0][34]}
        self.prediction = sorted(self.prediction.items(), key=operator.itemgetter(1), reverse=True)
        self.text2.config(text="Loaded ISL model from disk")

    def trsl(self):
        self.arcs = 'trsl'
        # Loading the model
        self.json_file = open("Models\TrSl\CNN\CNNThresholdTrSL.json", "r")
        self.model_json = self.json_file.read()
        self.json_file.close()
        self.loaded_model = model_from_json(self.model_json)
        # load weights into new model
        self.loaded_model.load_weights("Models\TrSL\CNN\CNNThresholdTrSL.h5")
        self.result = self.loaded_model.predict(self.test_image.reshape(1, 64, 64, 1))
        self.prediction = {'A': self.result[0][0], 'B': self.result[0][1], 
                    'C': self.result[0][2], 'D': self.result[0][3],
                    'E': self.result[0][4], 'F': self.result[0][5],
                    'G': self.result[0][6], 'H': self.result[0][7],
                    'I': self.result[0][8], 'J': self.result[0][9],
                    'K': self.result[0][10], 'L': self.result[0][11],
                    'M': self.result[0][12], 'N': self.result[0][13],
                    'O': self.result[0][14], 'P': self.result[0][15],
                    'R': self.result[0][16], 'S': self.result[0][17],
                    'T': self.result[0][18], 'U': self.result[0][19],
                    'V': self.result[0][20], 'Y': self.result[0][21],
                    'Z': self.result[0][22], 'del': self.result[0][23],
                    'nothing': self.result[0][24], 'space': self.result[0][25]}
        self.prediction = sorted(self.prediction.items(), key=operator.itemgetter(1), reverse=True)
        self.text2.config(text="Loaded TrSL model from disk")
    
    # def inception(self):
    #     # Loading the model
    #     self.json_file = open("cnn-aug_model.json", "r")
    #     self.model_json = self.json_file.read()
    #     self.json_file.close()
    #     self.loaded_model = model_from_json(self.model_json)
    #     # load weights into new model
    #     self.loaded_model.load_weights("CNN-Aug.h5")
    #     self.text2.config(text="Loaded inception model from disk")
    #     print("Loaded inception model from disk")
    
    # def cnn(self):
    #     # Loading the model
    #     self.json_file = open("cnn-aug_model.json", "r")
    #     self.model_json = self.json_file.read()
    #     self.json_file.close()
    #     self.loaded_model = model_from_json(self.model_json)
    #     # load weights into new model
    #     self.loaded_model.load_weights("CNN-Aug.h5")
    #     self.text2.config(text="Loaded cnn model from disk")
    #     print("Loaded cnn model from disk")
    
    # Function to add the predicted character to the word
    def add(self, p):
        if(p == 'del'):
            self.word = self.word[:-1]
        elif(p == 'nothing'):
            self.word = self.word
        elif(p == 'space'):
            # Auto correct the word
            self.word = self.spell.correction(self.word)

            # Text to speech
            if(len(self.word) != 0):
                # Convert the word to mp3
                myobj = gTTS(text=self.word, lang='en', slow=False)
                
                # Saving the converted audio in a mp3 file named
                # words 
                myobj.save("words.mp3")
                
                # Play the audio
                playsound('words.mp3', False)
                time.sleep(1)
                # Delete the file
                os.remove('words.mp3')

            # If the sentence is over 20 characters, go to next line
            if(len(self.sentence)>20):
                self.word = self.word + '\n'
            # Add the word to the sentence
            self.sentence = self.sentence + ' ' + self.word
            # Empty the word
            self.word = ' '
        else:
            self.word = self.word + p
    
    def video_stream(self):
        res, frame = self.cap.read()
        # Category dictionary
        # categories = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
        #                 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V',
        #                 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'del', 27: 'nothing', 28: 'space', 29: 'none', 30: 'none', 31: 'none', 32: 'none', 33: 'none', 34: 'none'}
        if res:
            # frame = cv2.flip(frame, 1)
            # Simulating mirror image
            cv2image  = cv2.flip(frame, 1)
            frame  = cv2.flip(frame, 1)
            
            # Coordinates of the ROI
            x1 = int(0.5*frame.shape[1])
            y1 = 10
            x2 = frame.shape[1]-10
            y2 = int(0.5*frame.shape[1])
            # Drawing the ROI
            # The increment/decrement by 1 is to compensate for the bounding box
            cv2.rectangle(cv2image, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
            cv2image  = cv2.cvtColor(cv2image , cv2.COLOR_BGR2RGBA)
            # Extracting the ROI
            # cv2image = cv2image[y1:y2, x1:x2]
            self.roi = frame[y1:y2, x1:x2]
            
            # Resizing the ROI so it can be fed to the model for self.prediction
            # print(self.roi.shape)
            self.roi = cv2.resize(self.roi, (64, 64))
            # self.roi.convert('L').show()
            # self.roi = np.expand_dims(self.roi, axis=3)
            # print(self.roi.shape)
            _, self.test_image = cv2.threshold(self.roi, 120, 255, cv2.THRESH_BINARY)
            
            minValue = 70
            gray = cv2.cvtColor(self.roi, cv2.COLOR_BGR2GRAY)
            # print(gray.shape)

            blur = cv2.GaussianBlur(gray,(5,5),2)
            th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
            res, self.test_image = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            self.test_image = cv2.resize(self.test_image, (64,64))
            self.test_image = cv2.flip(self.test_image, 1)

            # minValue = 70
            # gray = cv2.cvtColor(self.roi, cv2.COLOR_BGR2GRAY)
            # print(gray.shape)

            # blur = cv2.GaussianBlur(gray,(5,5),2)
            # th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
            # res, self.test_image = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            # self.test_image = cv2.resize(self.test_image, (300,300))
            # self.roi = np.expand_dims(self.test_image, axis=2)

            
            # Display the test image on tk
            newImage = cv2.resize(self.test_image, (300, 300))
            newImage = cv2.flip(newImage, 1)
            self.imgTests = Image.fromarray(newImage)

            imgtk = ImageTk.PhotoImage(master = self.root, image=self.imgTests)
            self.imgTest.imgtk = imgtk
            self.imgTest.configure(image=imgtk)

            # Display the main image on tk
            self.imgFrames = Image.fromarray(cv2image)

            imgtk2 = ImageTk.PhotoImage(master = self.root, image=self.imgFrames)
            self.imgFrame.imgtk = imgtk2
            self.imgFrame.configure(image=imgtk2)

            # print(self.roi.shape)
            
            # self.prediction = {'1': self.result[0][0], '2': self.result[0][1], 
            #         '3': self.result[0][2], '4': self.result[0][3],
            #         '5': self.result[0][4], '6': self.result[0][5],
            #         '7': self.result[0][6], '8': self.result[0][7],
            #         '9': self.result[0][8], 'A': self.result[0][9],
            #         'B': self.result[0][10], 'C': self.result[0][11],
            #         'D': self.result[0][12], 'E': self.result[0][13],
            #         'F': self.result[0][14], 'G': self.result[0][15],
            #         'H': self.result[0][16], 'I': self.result[0][17],
            #         'J': self.result[0][18], 'K': self.result[0][19],
            #         'L': self.result[0][20], 'M': self.result[0][21],
            #         'N': self.result[0][22], 'O': self.result[0][23],
            #         'P': self.result[0][24], 'Q': self.result[0][25],
            #         'R': self.result[0][26], 'S': self.result[0][27],
            #         'T': self.result[0][28], 'U': self.result[0][29],
            #         'V': self.result[0][30], 'W': self.result[0][31],
            #         'X': self.result[0][32], 'Y': self.result[0][33],
            #         'Z': self.result[0][34]}
            # Sorting based on top self.prediction
            if(self.arcs == 'asl'):
                self.result = self.loaded_model.predict(self.test_image.reshape(1, 64, 64, 1))
                self.prediction = {'A': self.result[0][0], 'B': self.result[0][1], 
                        'C': self.result[0][2], 'D': self.result[0][3],
                        'E': self.result[0][4], 'F': self.result[0][5],
                        'G': self.result[0][6], 'H': self.result[0][7],
                        'I': self.result[0][8], 'J': self.result[0][9],
                        'K': self.result[0][10], 'L': self.result[0][11],
                        'M': self.result[0][12], 'N': self.result[0][13],
                        'O': self.result[0][14], 'P': self.result[0][15],
                        'Q': self.result[0][16], 'R': self.result[0][17],
                        'S': self.result[0][18], 'T': self.result[0][19],
                        'U': self.result[0][20], 'V': self.result[0][21],
                        'W': self.result[0][22], 'X': self.result[0][23],
                        'Y': self.result[0][24], 'Z': self.result[0][25],
                        'del': self.result[0][26], 'nothing': self.result[0][27],
                        'space': self.result[0][28]}
                
                self.prediction = sorted(self.prediction.items(), key=operator.itemgetter(1), reverse=True)
            
            # At regular intervals add the predicted letter to the word
            if(self.i % 20 == 0):
                self.i = 0
                self.add(self.prediction[0][0])
            
            # Display the Character
            self.textch.config(text=self.prediction[0][0], font=("Courier",22,"bold"))
            # Display the Word
            self.textWord.config(text=self.word, font=("Courier",22,"bold"))
            # Display the Sentence
            self.textSentence.config(text=self.sentence, font=("Courier",22,"bold"))
            
            # Press esc to exit
            interrupt = cv2.waitKey(10)
            if interrupt & 0xFF == 27: # esc key
                exit()
            
            # repeat the called function ever 100msconda activate
            self.root.after(100, self.video_stream)
        self.i += 1
        
app = Vid()

app.root.mainloop()