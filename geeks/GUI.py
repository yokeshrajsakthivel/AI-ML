from tkinter import *
import numpy as np
from PIL import ImageGrab
from Prediction import predict

# Creating main window
window = Tk()
window.title("Handwritten Digit Recognition")
l1 = None  # Initialize l1 globally


def MyProject():
    global l1

    widget = cv
    # Getting the coordinates of the canvas
    x = window.winfo_rootx() + widget.winfo_x()
    y = window.winfo_rooty() + widget.winfo_y()
    x1 = x + widget.winfo_width()
    y1 = y + widget.winfo_height()

    # Capturing image from canvas and resizing it to (28x28) px
    img = ImageGrab.grab().crop((x, y, x1, y1)).resize((28, 28))

    # Converting the image to grayscale
    img = img.convert('L')

    # Extracting pixel matrix and flattening it to a (1, 784) vector
    vec = np.array(img).reshape(1, 784)

    # Normalize the pixel values (0-255) to (0-1)
    vec = vec / 255

    # Loading trained model weights (Theta1, Theta2)
    Theta1 = np.loadtxt('Theta1.txt')
    Theta2 = np.loadtxt('Theta2.txt')

    # Predict the digit using the trained model
    pred = predict(Theta1, Theta2, vec)

    # Remove the previous label if exists
    if l1:
        l1.destroy()

    # Display the predicted digit
    l1 = Label(window, text="Digit = " + str(pred[0]), font=('Algerian', 20))
    l1.place(x=230, y=420)


# Initialize global variables for mouse drawing
lastx, lasty = None, None


# Clears the canvas
def clear_widget():
    global cv, l1
    cv.delete("all")
    if l1:
        l1.destroy()


# Activates canvas for drawing
def event_activation(event):
    global lastx, lasty
    lastx, lasty = event.x, event.y
    cv.bind('<B1-Motion>', draw_lines)  # Bind mouse motion to draw_lines


# Draw lines on canvas
def draw_lines(event):
    global lastx, lasty
    x, y = event.x, event.y
    cv.create_line((lastx, lasty, x, y), width=30, fill='white', capstyle=ROUND, smooth=TRUE, splinesteps=12)
    lastx, lasty = x, y


# Title label
L1 = Label(window, text="Handwritten Digit Recognition", font=('Algerian', 25), fg="blue")
L1.place(x=35, y=10)

# Button to clear canvas
b1 = Button(window, text="Clear Canvas", font=('Algerian', 15), bg="orange", fg="black", command=clear_widget)
b1.place(x=120, y=370)

# Button to predict the digit drawn on canvas
b2 = Button(window, text="Prediction", font=('Algerian', 15), bg="white", fg="red", command=MyProject)
b2.place(x=320, y=370)

# Setting properties of the canvas
cv = Canvas(window, width=350, height=290, bg='black')
cv.place(x=120, y=70)
cv.bind('<Button-1>', event_activation)

# Set the size of the window
window.geometry("600x500")
window.mainloop()
