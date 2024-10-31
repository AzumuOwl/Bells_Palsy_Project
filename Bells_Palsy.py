import tkinter as tk
import subprocess
import os
import signal

# Initialize the main application window
root = tk.Tk()
root.title("Bell's Palsy Project")
root.geometry("600x300")
root.configure(bg="lightgray")

# Placeholder for the current process
current_process = None

def run_script(script_name):
    global current_process
    # Stop any running process before starting a new one
    stop_script()
    # Start a new script and save the process reference
    current_process = subprocess.Popen(['python', script_name])

def stop_script():
    global current_process
    if current_process:
        os.kill(current_process.pid, signal.SIGTERM)
        current_process = None

def close_program():
    stop_script()
    root.destroy()

# Title label
title_label = tk.Label(root, text="Bell's Palsy Project", font=("Helvetica", 16, "bold"), bg="lightgray")
title_label.pack(pady=10)

# Frame for main action buttons
button_frame = tk.Frame(root, bg="lightgray")
button_frame.pack(pady=10)

# Define labels and scripts for each button
button_labels = ["ยักคิ้ว", "ขมวดคิ้ว", "ย่นจมูก", "กระพริบตา", "ทำปากจู๋"]
button_scripts = ["1.py", "2.py", "3.py", "4.py", "5.py"]

# Create action buttons with custom labels
for index, (label, script) in enumerate(zip(button_labels, button_scripts)):
    button = tk.Button(
        button_frame, text=label,
        font=("Helvetica", 12),
        width=10, height=2,
        command=lambda script=script: run_script(script)
    )
    button.grid(row=0, column=index, padx=10, pady=5)

# Frame for control buttons (Stop and Close)
control_frame = tk.Frame(root, bg="lightgray")
control_frame.pack(pady=20)

# Button to stop the running script
stop_button = tk.Button(
    control_frame, text="Stop Script",
    font=("Helvetica", 12),
    width=12, height=2,
    command=stop_script,
    bg="lightcoral"
)
stop_button.grid(row=0, column=0, padx=10, pady=5)

# Button to close the program
close_button = tk.Button(
    control_frame, text="Close Program",
    font=("Helvetica", 12),
    width=12, height=2,
    command=close_program,
    bg="lightblue"
)
close_button.grid(row=0, column=1, padx=10, pady=5)

# Run the Tkinter main loop
root.mainloop()
