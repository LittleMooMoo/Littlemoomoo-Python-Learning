# Lesson 1: Python Basics and Motor Safety

## Big Idea

Python syntax is not just grammar. In robotics, syntax becomes a way to express
engineering rules:

- What command did I send?
- Is this command safe?
- What should the robot do if the command is too large?
- How do I print useful debug information?

In this lesson, you will learn:

- variables
- basic data types
- `if`, `elif`, and `else`
- functions
- return values
- simple debug printing
- why safety checks matter in robot code

## Vocabulary

`variable`
: A name that stores a value.

`float`
: A number with decimals, such as `0.25` or `-1.0`.

`bool`
: A true/false value: `True` or `False`.

`function`
: A reusable block of code that can accept inputs and return an output.

`clamp`
: To limit a value so it stays inside a safe range.

## Robotics Connection

Imagine your robot code calculates a motor command of `1.4`, but your motor
controller expects values only between `-1.0` and `1.0`.

Sending `1.4` directly is unsafe. A real robot program should check the command
before sending it to hardware.

So we write a function:

```python
def clamp_motor_command(command):
    if command > 1.0:
        return 1.0
    elif command < -1.0:
        return -1.0
    else:
        return command
```

This is beginner Python, but it is also real engineering logic.

## Mental Model

Think of a function like a small machine:

```text
input command -> safety logic -> safe output command
```

If the input is too high, the function returns the maximum safe value.
If the input is too low, the function returns the minimum safe value.
If the input is already safe, the function returns it unchanged.

## Run the Demo

From the repo root:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe" 01_python_basics\motor_safety_demo.py
```

Read the output slowly. Try to predict each safe command before looking at the
printed result.

## Practice Questions

1. What is the difference between `command` and `safe_command`?
2. Why does the code use `elif` instead of a second `if`?
3. What would happen if the safe range were `-0.5` to `0.5` instead?
4. Why is printing debug information useful before controlling real hardware?

## Your First Modification

Open `exercise_01_motor_safety.py`.

Try changing:

```python
MAX_COMMAND = 1.0
MIN_COMMAND = -1.0
```

to:

```python
MAX_COMMAND = 0.6
MIN_COMMAND = -0.6
```

Then run:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe" 01_python_basics\exercise_01_motor_safety.py
```

Predict the output before running it.
