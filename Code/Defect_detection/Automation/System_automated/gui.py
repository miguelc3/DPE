from tkinter import *
from PIL import ImageTk, Image
import datetime

root = Tk()

# Title and icon
root.title('Defect detection on painted surfaces')
root.iconbitmap('logo.ico') 

button_exit = Button(root, text="EXIT", command=root.quit)
my_img = ImageTk.PhotoImage(Image.open("falta_tinta.png").resize((808, 608), Image.ANTIALIAS))
my_label_img = Label(image=my_img)
my_label_class = Label(root, text="Clasification: FALTA DE TINTA", bg="red", fg="white")
my_label_total = Label(root, text="Total parts: 2")

currentDate = datetime.datetime.now()
year = str(currentDate.year)
month = str(currentDate.month)
day = str(currentDate.day)
hour = str(currentDate.hour)
min = str(currentDate.minute)
sec = str(currentDate.second)

text = day + "/" + month + "/" + year + " - " + hour + ":" + min
status = Label(root, text=text, bd=1, relief=SUNKEN, anchor=E)

identifier = '8370_' + year + '_' + month + '_' + day + '_' + hour + '_' + min + '_' + sec + '_999999999'
my_label_part = Label(root, text="Identifier: " + identifier, bg="white")

my_label_part.grid(row=0, column=0)
button_exit.grid(row=2, column=2, padx=15, pady=10)
my_label_img.grid(row=1, column=0, columnspan=3)
my_label_class.grid(row=2, column=0)
my_label_total.grid(row=2, column=1)
status.grid(row=3, column=0, columnspan=3, sticky=W+E)

root.mainloop()
