import tkinter as tk


def get_screen_size():

    root = tk.Tk()

    root.withdraw()

    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()

    root.destroy()

    return width, height


if __name__ == "__main__":

    width, height = get_screen_size()

    print(
        f"Screen: {width} x {height}"
    )