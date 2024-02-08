import tkinter as tk
from tkinter import messagebox


def button_click(button_number):
    textbox.insert(tk.END, f"Button {button_number} clicked!\n")


def quit_app():
    if messagebox.askokcancel("Quit", "Do you really want to quit?"):
        root.destroy()


# Create the main window
root = tk.Tk()
root.title("K-STEAM TEST APP")
# Set size of the main window
root.geometry("400x250")
# Set the minimum size of the main window
root.minsize(500, 250)
# Set the maximum size of the main window
root.maxsize(1000, 500)
# Set the background color of the main window
root.config(bg="white")


# Create a textbox
textbox = tk.Text(root, height=10, width=40)
textbox.pack(pady=10)
# Set the initial text of the textbox
textbox.insert(tk.END, "Welcome to K-STEAM!\n")
# Disable editing of the textbox
# textbox.config(state=tk.DISABLED)
# Set the background color of the textbox
textbox.config(bg="blue")
# Set the text color of the textbox
textbox.config(fg="white")
# Set the font of the textbox
textbox.config(font=("Arial", 10))
# Set the border of the textbox
textbox.config(bd=2)
# Set the border color of the textbox
textbox.config(highlightbackground="red")
# Set the border width of the textbox
textbox.config(highlightthickness=2)
# set the padding of the textbox
textbox.config(padx=10, pady=5)

button_name = ["FPS", "Acc Motion Recognition", "Acc Skeleton Position"]

# Create three buttons
for i in range(1, 4):
    # get the text of the button
    btnName = button_name[i-1]
    button = tk.Button(
        root, text=f"{btnName}", command=lambda i=i: button_click(i))
    # Set the background color of the button
    button.config(bg="white")
    # Set the text color of the button
    button.config(fg="black")
    # Set the font of the button
    button.config(font=("Arial", 10, "bold"))
    # Set the border of the button
    button.config(bd=2)
    # Set the border color of the button
    button.config(highlightbackground="red")
    # Set the border width of the button
    button.config(highlightthickness=2)
    # Set the padding of the button
    button.config(padx=2, pady=2)
    # Set the width of the button
    
    button.pack(side=tk.LEFT, padx=5)

# Create a quit button
quit_button = tk.Button(root, text="Quit", command=quit_app)
quit_button.config(bg="white")
# Set the text color of the button
quit_button.config(fg="black")
# Set the font of the button
quit_button.config(font=("Arial", 10, "bold"))
# Set the border of the button
quit_button.config(bd=2)
# Set the border color of the button
quit_button.config(highlightbackground="red")
# Set the border width of the button
quit_button.config(highlightthickness=2)
# Set the padding of the button
quit_button.config(padx=2, pady=2)
quit_button.pack(side=tk.RIGHT, padx=5)

# Run the main loop
root.mainloop()
