import os
import tkinter as tk
from tkinter import filedialog
from tkinterdnd2 import TkinterDnD, DND_FILES
from PIL import Image, ImageTk
from src.backend import colorize  # Assuming colorize is the function you want to call

def main():
    root = TkinterDnD.Tk()
    app = ColorApp(root)
    root.mainloop()

class ColorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Folder Path App")
        self.root.geometry("950x540")  # Adjust the window size as needed
        self.background_image_path = "./illustration/background.png"
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
        browse_button = tk.Button(root, text="Browse", command=self.browse_image)
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
                                text="Drag & Drop any image here !",
                                font=("Calibri", 25), fill="white", anchor="center")

        text_position = self.original_image.width// 2 , self.original_image.height // 1.7
        self.canvas.create_text(text_position,
                                text="Find your colorized images in the image folder",
                                font=("Calibri", 18), fill="white", anchor="center")


        # Bind the drag-and-drop events using tkinterdnd2
        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind('<<Drop>>', self.on_drop)

    def browse_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp")])
        if file_path:
            self.folder_path_entry.delete(0, tk.END)
            self.folder_path_entry.insert(0, file_path)
            print(file_path)
            colorize(file_path)
            self.show_images()

    def on_drop(self, event):
        file_path = event.data
        file_path = file_path.replace('{', '')
        file_path = file_path.replace('}', '')
        print(file_path)
        if file_path and file_path.endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
            # self.folder_path_entry.delete(0, tk.END)
            # self.folder_path_entry.insert(0, file_path)
            colorize(file_path)
            self.show_images()
            # pass

    def is_image(self, file_path):
        try:
            with Image.open(file_path):
                return True
        except (IOError, SyntaxError):
            return False

    def show_images(self):
        paths = ["./images/model_1/", "./images/model_2/", "./images/model_3/", "./images/gray/", "./images/provided/img/"]

        image_list = self.load_images(paths)

        for i, image in enumerate(image_list):
            second_window = tk.Toplevel(self.root)
            second_window.title(paths[i].split('/')[-2])

            label = tk.Label(second_window)
            label.pack()

            label.config(image=image)
            label.image = image
            second_window.update()
            second_window.after(10000)  # Display each image for 2 seconds (adjust as needed)

    def load_images(self, paths):
        image_list = []
        image_name = None

        files = [f for f in os.listdir(paths[0])]
        if not files:
            return None

        full_paths = [os.path.join(paths[0], f) for f in files]
        latest_file = max(full_paths, key=os.path.getmtime)
        image_name = latest_file.split('/')[-1].split('_')[0]

        for path in paths:
            for filename in os.listdir(path):
                if filename.split('_')[0] == image_name:
                    file_path = os.path.join(path, filename)
                    image = Image.open(file_path)
                    photo = ImageTk.PhotoImage(image)
                    image_list.append(photo)
        return image_list