import cv2
import os
from ultralytics import YOLO
import scrython


# Class for Magic The Gathering cards to hold name, manacost, card type, CMC value, etc.
class mtgCard:
    def __init__(self, name, manaCost, cardType):
        self.name = name
        self.manaCost = manaCost
        self.cardType = cardType
        self.cmc = self.calcCMC()

    def __str__(self):
        return f"{self.name} [{self.cardType}] ({self.cmc})"

    def calcCMC (self):
        # CMC value to be calculated
        cmc = 0
        # Colorless mana value of card, if there is any
        colorlessMana = ""
        # loop through "manacost" string and count the amount of characters to determine mana value
        for char in self.manaCost:
            # check if the char is a mana type (F,W,U,R,G) or generic mana cost
            if char.isalpha():
                cmc += 1
            elif char.isnumeric():
                colorlessMana = colorlessMana + char
        # check if colorless mana was added to the object
        # if colorless mana was added, the string of colorlessMana is turned into an integer and added to cmc value
        if len(colorlessMana) > 0:
            cmc += int(colorlessMana)

        return cmc


def cam():
    cam = cv2.VideoCapture(0)  # video capture object
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # set width of video window
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # set height of video window

    path = r'C:\Users\thele\runs\detect\train2'
    model_path = os.path.join(path, 'weights', 'last.pt')
    model = YOLO(model_path)  # use trained model from designated path
    threshold = 0.65  # accepted threshold of recognized object to be shown

    cardname = ''

    # loop captured frame so a video is displayed
    while True:
        ret, image = cam.read()  # capture video frame

        results = model(image, verbose=False)[0]
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if score > threshold:
                cv2.rectangle(image, (int(x1), int(y1), int(x2), int(y2)), (0, 255, 0), 4)
                cv2.putText(image, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
                cardname = results.names[int(class_id)]
        # display captured frame
        cv2.imshow("random", image)
        if cv2.waitKey(1) == ord('q'):  # hit 'q' button within captured frame window to quit
            break
        elif cv2.waitKey(1) == ord('r'):
            card = scrython.cards.Named(fuzzy=cardname)
            c1 = mtgCard(card.name(), card.mana_cost(), card.type_line())
            print(c1)


def yolotrain():
    # Load a model
    model = YOLO("yolov8n.yaml")  # build a new model from scratch

    # Use model
    results = model.train(data='config.yaml', epochs=100)  # train the model


if __name__ == '__main__':
    cam()
