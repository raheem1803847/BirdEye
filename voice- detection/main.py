import speech_recognition as sr
import os


fileName = input("give a file name: ") + ".txt"
r = sr.Recognizer()

with sr.Microphone() as source:
    print('Say Something:')
    audio = r.listen(source)
    print ('Done!')


    text = r.recognize_google(audio, language='ar-EG')
    print(text)


myFile = open(fileName, 'w',encoding="utf-8")
myFile.write(text)
myFile.close()

os.startfile(fileName)