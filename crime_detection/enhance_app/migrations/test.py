import tkinter as tk
import math


class AdvancedCalculatorGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Advanced Calculator")

        # Create the entry widget to display the current value
        self.entry = tk.Entry(self.root, width=30)
        self.entry.grid(row=0, column=0, columnspan=5)

        # Create the buttons for the different operations
        self.create_button("+", self.add, 1, 0)
        self.create_button("-", self.subtract, 1, 1)
        self.create_button("*", self.multiply, 1, 2)
        self.create_button("/", self.divide, 1, 3)
        self.create_button("^", self.exponent, 1, 4)

        self.create_button("sqrt", self.square_root, 2, 0)
        self.create_button("sin", self.sine, 2, 1)
        self.create_button("cos", self.cosine, 2, 2)
        self.create_button("tan", self.tangent, 2, 3)
        self.create_button("factorial", self.factorial, 2, 4)

        self.create_button("clear", self.clear, 3, 0)
        self.create_button("quit", self.root.destroy, 3, 4)

        self.current_value = 0

        self.root.mainloop()

    def create_button(self, text, command, row, column):
        button = tk.Button(self.root, text=text, command=command)
        button.grid(row=row, column=column)

    def add(self):
        operand = float(self.entry.get())
        self.current_value += operand
        self.entry.delete(0, tk.END)
        self.entry.insert(0, str(self.current_value))

    def subtract(self):
        operand = float(self.entry.get())
        self.current_value -= operand
        self.entry.delete(0, tk.END)
        self.entry.insert(0, str(self.current_value))

    def multiply(self):
        operand = float(self.entry.get())
        self.current_value *= operand
        self.entry.delete(0, tk.END)
        self.entry.insert(0, str(self.current_value))

    def divide(self):
        operand = float(self.entry.get())
        self.current_value /= operand
        self.entry.delete(0, tk.END)
        self.entry.insert(0, str(self.current_value))

    def exponent(self):
        operand = float(self.entry.get())
        self.current_value = self.current_value ** operand
        self.entry.delete(0, tk.END)
        self.entry.insert(0, str(self.current_value))

    def square_root(self):
        result = math.sqrt(self.current_value)
        self.entry.delete(0, tk.END)
        self.entry.insert(0, str(result))

    def sine(self):
        result = math.sin(self.current_value)
        self.entry.delete(0, tk.END)
        self.entry.insert(0, str(result))

    def cosine(self):
        result = math.cos(self.current_value)
        self.entry.delete(0, tk.END)
        self.entry.insert(0, str(result))

    def tangent(self):
        result = math.tan(self.current_value)
        self.entry.delete(0, tk.END)
        self.entry.insert(0, str(result))

    def factorial(self):
        result = math.factorial(int(self.current_value))
        self.entry.delete(0, tk.END)
        self.entry.insert(0, str(result))

    def clear(self):
        self.current_value = 0
        self.entry.delete(0, tk.END)


if __name__ == "__main__":
    calculator = AdvancedCalculatorGUI()
