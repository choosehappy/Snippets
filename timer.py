import tkinter as tk
import time

class TimerApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.overrideredirect(True)  # No title bar
        self.root.attributes("-topmost", True)  # Always on top
        self.root.configure(bg="black")
        self.root.geometry("+100+100")  # Starting position

        # Create label
        #self.label = tk.Label(self.root, text="00:00", font=("Arial", 32), fg="white", bg="black")
        self.label = tk.Label(self.root, text="00:00", font=("Arial", 16), fg="black", bg="white")
        self.label.pack()

        # Right-click menu
        self.menu = tk.Menu(self.root, tearoff=0)
        self.menu.add_command(label="Start", command=self.start)
        self.menu.add_command(label="Stop", command=self.stop)
        self.menu.add_command(label="Reset", command=self.reset)

        # Bind right click to open menu
        self.label.bind("<Button-3>", self.show_menu)

        # Bind drag to move window
        self.label.bind("<Button-1>", self.start_move)
        self.label.bind("<B1-Motion>", self.do_move)

        # Timer variables
        self.running = False
        self.start_time = 0
        self.elapsed = 0

        # Start timer update loop
        self.update_timer()

        self.root.mainloop()

    def show_menu(self, event):
        self.menu.post(event.x_root, event.y_root)

    def start_move(self, event):
        self._x = event.x
        self._y = event.y

    def do_move(self, event):
        x = self.root.winfo_x() + event.x - self._x
        y = self.root.winfo_y() + event.y - self._y
        self.root.geometry(f"+{x}+{y}")

    def start(self):
        if not self.running:
            self.running = True
            self.start_time = time.time() - self.elapsed

    def stop(self):
        if self.running:
            self.running = False
            self.elapsed = time.time() - self.start_time

    def reset(self):
        self.running = False
        self.elapsed = 0
        self.label.config(text="00:00")

    def update_timer(self):
        if self.running:
            self.elapsed = time.time() - self.start_time
            minutes = int(self.elapsed // 60)
            hours = int(minutes // 60)
            self.label.config(text=f"{hours:02d}:{minutes % 60:02d}")
        self.root.after(1000, self.update_timer)

# Run the app
TimerApp()
