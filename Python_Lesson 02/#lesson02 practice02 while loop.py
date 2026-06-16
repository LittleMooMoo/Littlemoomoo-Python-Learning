# lesson02 practice02 while loop
import math

while True:
    target = input("Enter a number to start the loop, or type q to quit:")
    if target == "q":
        break
    if target == "pi":
        target = math.pi
    elif target == "e":
        target = math.e
    try:
        input_target = float(target)
    except ValueError:
        input_target = None
        print("Please enter a valid number or q to quit.")
    if input_target is not None:
        new_target = input_target % 3
        if math.isclose(new_target, 0):
            new_target = input_target / 2
            print("Number Shrunk:", new_target)
        else:
            new_target = input_target * math.e
            print("Number Expanded:", new_target)
