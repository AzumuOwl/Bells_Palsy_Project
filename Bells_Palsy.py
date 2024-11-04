import tkinter as tk
import subprocess
import os
import signal
import pygame  # Import pygame for sound

# Initialize pygame mixer for playing sound
pygame.mixer.init()

# Initialize the main application window
root = tk.Tk()
root.title("Bell's Palsy Project")
root.geometry("800x600")  # เพิ่มขนาดหน้าต่าง
root.configure(bg="lightgray")

# Placeholder for the current process
current_process = None

def notified1():
    """Load and play sound on button press."""
    pygame.mixer.music.load('S1/notified3.mp3')  # Replace with your actual file path
    pygame.mixer.music.play()

def run_script(script_name):
    """Stop any running process before starting a new one."""
    global current_process
    stop_script()  # Ensure no other script is running
    # Start the new script and save the process reference
    current_process = subprocess.Popen(['python', script_name])

def stop_script():
    """Terminate the running script."""
    global current_process
    if current_process:
        os.kill(current_process.pid, signal.SIGTERM)
        current_process = None

def close_program():
    """Stop any running script and close the program."""
    stop_script()
    root.destroy()

# Title label
title_label = tk.Label(root, text="Bell's Palsy Project", font=("Helvetica", 18, "bold"), bg="lightgray")
title_label.pack(pady=15)

# Frame for main action buttons
button_frame = tk.Frame(root, bg="lightgray")
button_frame.pack(pady=15)

# Define labels and scripts for each button
button_labels = [
    "ยักคิ้ว", "ขมวดคิ้ว", "ย่นจมูก", "กระพริบตา",
    "ทำปากจู๋", "ทำจมูกบาน", "ท่ายิ้มไม่ยกมุมปาก", "ท่ายิ้มแสดงฟัน"
]
button_scripts = ["1.py", "2.py", "3.py", "4.py", "5.py", "6.py", "7.py", "8.py"]

# Create action buttons with custom labels
for index, (label, script) in enumerate(zip(button_labels, button_scripts)):
    button = tk.Button(
        button_frame, text=label,
        font=("Helvetica", 14),  # เพิ่มขนาดฟอนต์
        width=12, height=3,  # เพิ่มขนาดปุ่ม
        command=lambda script=script: run_script(script),
    )
    button.grid(row=index // 4, column=index % 4, padx=15, pady=10)  # เพิ่มระยะห่างระหว่างปุ่ม

# Frame for control buttons (Stop and Close)
control_frame = tk.Frame(root, bg="lightgray")
control_frame.pack(pady=20)

# Stop script button
stop_button = tk.Button(
    control_frame, text="Stop Script",
    font=("Helvetica", 14),
    width=15, height=3,  # เพิ่มขนาดปุ่ม
    command=stop_script,
    bg="lightcoral"
)
stop_button.grid(row=0, column=0, padx=15, pady=10)

# Close program button
close_button = tk.Button(
    control_frame, text="Close Program",
    font=("Helvetica", 14),
    width=15, height=3,  # เพิ่มขนาดปุ่ม
    command=close_program,
    bg="lightblue"
)
close_button.grid(row=0, column=1, padx=15, pady=10)

# Alert sound button
alert_button = tk.Button(
    control_frame, text="Alert!",
    font=("Helvetica", 14),
    width=15, height=3,  # เพิ่มขนาดปุ่ม
    command=notified1,
    bg="lightgreen"
)
alert_button.grid(row=0, column=2, padx=15, pady=10)

# Run the Tkinter main loop
root.mainloop()
