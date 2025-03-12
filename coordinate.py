import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class PolygonROISelector:
    def __init__(self, master, image_path, target_size=(640, 480), required_points=4, required_polygons=4):
        self.master = master
        self.target_size = target_size
        self.required_points = required_points  # Number of points per polygon (e.g., 4 for a quadrilateral)
        self.required_polygons = required_polygons  # Total number of polygons to draw
        
        # Load and resize the image
        self.orig_image = Image.open(image_path)
        self.resized_image = self.orig_image.resize(self.target_size, Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(self.resized_image)
        
        # Create a canvas to display the image
        self.canvas = tk.Canvas(master, width=self.tk_image.width(), height=self.tk_image.height())
        self.canvas.pack()
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        
        # Variables to store points and polygons
        self.current_points = []  # Points for the current polygon being drawn
        self.polygons = []  # List of all drawn polygons
        
        # Bind the left mouse button click event
        self.canvas.bind("<Button-1>", self.on_click)
        
        # A label to give instructions to the user
        self.info_label = tk.Label(master, text="Click to define a polygon (4 points per polygon).")
        self.info_label.pack(pady=10)
    
    def on_click(self, event):
        # Append the clicked point to the current polygon list
        self.current_points.append((event.x, event.y))
        # Draw a small circle at the clicked location for visual feedback
        r = 3
        self.canvas.create_oval(event.x - r, event.y - r, event.x + r, event.y + r, fill="blue")
        
        # If the required number of points have been collected, draw the polygon
        if len(self.current_points) == self.required_points:
            # Draw the polygon on the canvas with a red outline
            self.canvas.create_polygon(self.current_points, outline="red", fill="", width=2)
            # Save the polygon's coordinates
            self.polygons.append(self.current_points.copy())
            # Clear the current points for the next polygon
            self.current_points = []
            
            # Update instructions to the user
            if len(self.polygons) < self.required_polygons:
                self.info_label.config(text=f"{len(self.polygons)} polygon(s) drawn. Click to define the next polygon.")
            else:
                self.info_label.config(text="All 4 polygons drawn. Check the console for coordinates.")
                print("Coordinates for the 4 polygons:")
                for idx, poly in enumerate(self.polygons, start=1):
                    print(f"Polygon {idx}: {poly}")

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Polygon ROI Selection Tool")
    
    # Open a file dialog to select an image
    file_path = filedialog.askopenfilename(
        title="Select an Image", 
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.gif")]
    )
    
    if file_path:
        # Create the ROI selector with the desired target size
        selector = PolygonROISelector(root, file_path, target_size=(640, 480), required_points=4, required_polygons=4)
        root.mainloop()
    else:
        print("No image was selected.")