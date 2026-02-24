import tkinter as tk
from PIL import ImageGrab
import os
from datetime import datetime
import threading
import time


class ScreenshotApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.overrideredirect(True)  # No title bar
        self.root.attributes("-topmost", True)  # Always on top
        self.root.configure(bg="white")
        self.root.geometry("30x30+100+100")  # Starting position (narrow/small window)
        
        # Create canvas for drawing circle
        self.canvas = tk.Canvas(self.root, width=30, height=30, bg="white", highlightthickness=0)
        self.canvas.pack()
        
        # Draw red circle (active/recording)
        self.circle = self.canvas.create_oval(5, 5, 25, 25, fill="red", outline="")
        
        # Right-click menu
        self.menu = tk.Menu(self.root, tearoff=0)
        self.menu.add_command(label="Stop", command=self.stop_app)
        self.menu.add_command(label="Close", command=self.close_app)
        
        # Bind right click to open menu
        self.canvas.bind("<Button-3>", self.show_menu)
        
        # Bind drag to move window
        self.canvas.bind("<Button-1>", self.start_move)
        self.canvas.bind("<B1-Motion>", self.do_move)
        
        # Screenshot variables
        self.running = True
        self.screenshot_thread = None
        
        # Start screenshot thread
        self.start_screenshot_thread()
        
        self.root.mainloop()
    
    def show_menu(self, event):
        """Show the right-click menu"""
        self.menu.post(event.x_root, event.y_root)
    
    def start_move(self, event):
        """Record the starting position for dragging"""
        self.x = event.x
        self.y = event.y
    
    def do_move(self, event):
        """Move the window"""
        deltax = event.x - self.x
        deltay = event.y - self.y
        x = self.root.winfo_x() + deltax
        y = self.root.winfo_y() + deltay
        self.root.geometry(f"+{x}+{y}")
    
    def set_circle_color(self, color):
        """Change the circle color"""
        self.canvas.itemconfig(self.circle, fill=color)
    
    def take_screenshot(self):
        """Take a screenshot and save it to c:\\tmp\\[date]"""
        try:
            # Get current date for folder name
            date_str = datetime.now().strftime("%Y-%m-%d")
            folder_path = f"c:\\tmp\\screenshoter\\{date_str}"
            
            # Create folder if it doesn't exist
            os.makedirs(folder_path, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%H-%M-%S")
            filename = f"screenshot_{timestamp}.png"
            filepath = os.path.join(folder_path, filename)
            
            # Take screenshot
            screenshot = ImageGrab.grab()
            screenshot.save(filepath)
            
            # Show green circle for 1 second
            self.set_circle_color("green")
            self.root.after(1000, lambda: self.set_circle_color("red" if self.running else "blue"))
            
        except Exception as e:
            print(f"Error taking screenshot: {e}")
    
    def screenshot_loop(self):
        """Background thread that takes screenshots every 5 minutes"""
        while self.running:
            self.take_screenshot()
            # Wait 5 minutes (300 seconds)
            for _ in range(300):
                if not self.running:
                    break
                time.sleep(1)
    
    def start_screenshot_thread(self):
        """Start the background screenshot thread"""
        self.running = True
        self.screenshot_thread = threading.Thread(target=self.screenshot_loop, daemon=True)
        self.screenshot_thread.start()
        self.circle = self.canvas.create_oval(5, 5, 25, 25, fill="red", outline="")
        self.menu.entryconfigure(0, label="Stop",command=self.stop_app)
        
    def close_app(self):
        """Close the application"""
        self.running = False
        self.root.quit()
        self.root.destroy()

    def stop_app(self):
        """Close the application"""
        self.running = False
        self.circle = self.canvas.create_oval(5, 5, 25, 25, fill="blue", outline="")
        self.menu.entryconfigure(0, label="Start",command=self.start_screenshot_thread)
        
        
        

if __name__ == "__main__":
    app = ScreenshotApp()
