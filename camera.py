import cv2
import smtplib
import numpy as np
from model import FacialExpressionModel

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.model = FacialExpressionModel("model.json", "model_weights.h5")
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.sad_counter = 0
        self.angry_counter = 0

    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        _, fr = self.video.read()
        gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = self.facec.detectMultiScale(gray_fr, 1.3, 5)

        for (x, y, w, h) in faces:
            fc = gray_fr[y:y+h, x:x+w]

            roi = cv2.resize(fc, (48, 48))
            pred = self.model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])

            # add conditional statement to pop up red alarm for sad face
            if pred == "Sad":
                self.sad_counter += 1
                cv2.rectangle(fr, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(fr, "SAD", (x, y-10), self.font, 1, (0, 0, 255), 2)
                if self.sad_counter == 20:
                    print("Hello/n \n"*30)
                        # Email Generation
                    subject = "EmoDect Alert!"  # Subject of the Email
                    body = "One of your student is sad, Kindly check and report the status."  # Messae of the Email
                    msg = f'Subject: {subject}\n\n{body}'
                    server = smtplib.SMTP("smtp.gmail.com", 587)
                    server.starttls()
                    # Setting the Login Access
                    server.login("harisfazal678@gmail.com", "zrryklqbvzfwchzi")
                    #server.login("Email", "Applcaiton Token from Gmail")
                    

                    server.sendmail("harisfazal678@gmail.com", "harisfazal677@gmail.com", msg)  # Sending the Email
                    self.sad_counter = 0

        
                    
            else:
                cv2.rectangle(fr, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(fr, pred, (x, y), self.font, 1, (255, 255, 0), 2)

        # display sad face counter
        cv2.putText(fr, f"Sad Faces Detected: {self.sad_counter}", (50, 50), self.font, 1, (0, 0, 225), 2)

        _, jpeg = cv2.imencode('.jpg', fr)
        return jpeg.tobytes()


