import sys
import os

# from src.frontend import main


import tkinter as tk
from tkinter import filedialog
from tkinterdnd2 import TkinterDnD, DND_FILES
from PIL import Image, ImageTk
# from src.backend import colorize  # Assuming colorize is the function you want to call

def main():
    root = TkinterDnD.Tk()
    app = ColorApp(root)
    root.mainloop()

class ColorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Folder Path App")
        self.root.geometry("950x540")  # Adjust the window size as needed
        self.background_image_path = "../illustration/background.png"
        self.original_image = Image.open(self.background_image_path).resize((1000, 562), Image.LANCZOS)
        self.background_image = ImageTk.PhotoImage(self.original_image)
        self.root.resizable(False, False)

        # Create a canvas instead of label
        self.canvas = tk.Canvas(root, width=self.original_image.width, height=self.original_image.height)
        self.canvas.place(relx=0.5, rely=0.5, anchor="center")

        # Use create_image method to display the background image on the canvas
        self.canvas.create_image(0, 0, anchor="nw", image=self.background_image)

        # Create entry widget for folder path
        self.folder_path_entry = tk.Entry(root)

        # Create browse button
        browse_button = tk.Button(root, text="Browse", command=self.browse_folder)
        browse_button.place(relx=0.5, rely=0.85, anchor="center")

        # Add transparent text with background using create_text method
        text_position = self.original_image.width // 2, self.original_image.height // 10
        self.canvas.create_text(text_position, text="LANDSCAPE COLORIZATION", font=("Calibri", 54), fill="white", anchor="center")

        # Add transparent text with background using create_text method
        text_position = self.original_image.width * 6 // 8, self.original_image.height // 5
        self.canvas.create_text(text_position,
                                text="194.077 Applied Deep Learning WS2023\n"
                                     "             Grégoire de Lambertye 1220221",
                                font=("Calibri", 12), fill="white", anchor="center")

        # Add instructions
        text_position = self.original_image.width// 2 , self.original_image.height // 2
        self.canvas.create_text(text_position,
                                text="Drag & Drop any folder here !",
                                font=("Calibri", 25), fill="white", anchor="center")

        # Bind the drag-and-drop events using tkinterdnd2
        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind('<<Drop>>', self.on_drop)

    def browse_folder(self):
        file_or_folder_path = filedialog.askdirectory()
        if file_or_folder_path:
            self.folder_path_entry.delete(0, tk.END)
            self.folder_path_entry.insert(0, file_or_folder_path)
            # colorize(file_or_folder_path)

    def on_drop(self, event):
        file_path = event.data
        if file_path:
            self.folder_path_entry.delete(0, tk.END)
            self.folder_path_entry.insert(0, file_path)
            # colorize(file_path)





if getattr(sys, 'frozen', False):
    script_dir = os.path.dirname(sys.executable)
else:
    script_dir = os.path.dirname(os.path.realpath(__file__))

sys.path.append(os.path.join(script_dir, 'src'))

os.chdir(script_dir)
main()
