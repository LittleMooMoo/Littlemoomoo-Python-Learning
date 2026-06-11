# Beginner C++ And Python Loops Guide

This guide is written for a new undergrad who is learning programming from the
beginning, but wants the ideas to connect to robotics.

## What Is C++?

C++ is a fast, compiled programming language.

Python usually runs your `.py` file directly through the Python interpreter.
C++ usually needs a compiler first. The compiler translates your `.cpp` source
file into an executable program, such as `.exe` on Windows.

Simple mental model:

```text
Python:
    source code.py -> Python interpreter -> program runs

C++:
    source code.cpp -> compiler -> executable.exe -> program runs
```

Robotics often uses C++ because it is fast and gives the programmer more
control over memory, timing, and hardware-level behavior. Python is excellent
for learning, scripting, plotting, simulation, machine learning, and quick
experiments. Many robotics systems use both.

## How To Create A C++ File In VS Code

1. Open VS Code.
2. Open your project folder.
3. In the file explorer, create a new file.
4. Name it with the `.cpp` extension, for example:

```text
hello_robot.cpp
```

5. Write C++ code in that file.
6. Install the Microsoft C/C++ extension for VS Code.
7. Install a C++ compiler, such as MSVC or MinGW-w64.
8. Compile the file.
9. Run the compiled program.

Important: VS Code is an editor. It does not automatically include a C++
compiler. You need both:

```text
VS Code editor + C++ extension + C++ compiler
```

Official Microsoft/VS Code references:

- [C/C++ for Visual Studio Code](https://code.visualstudio.com/docs/languages/cpp)
- [Configure VS Code for Microsoft C++](https://code.visualstudio.com/docs/cpp/config-msvc)
- [Microsoft C++ documentation](https://learn.microsoft.com/cpp/)

## Python Loop Concepts

### `for` Loop

A `for` loop repeats code for each item in a group.

```python
motor_ids = [1, 2, 3]

for motor_id in motor_ids:
    print(motor_id)
```

Meaning:

```text
Take each motor_id from the list and run the indented code.
```

### `while` Loop

A `while` loop repeats as long as a condition is true.

```python
count = 0

while count < 3:
    print(count)
    count = count + 1
```

Meaning:

```text
Keep looping while count is less than 3.
```

### `break`

`break` exits a loop immediately.

```python
for motor_id in [1, 2, 3]:
    if motor_id == 2:
        break
```

### `continue`

`continue` skips the rest of the current loop iteration and moves to the next
item.

```python
for motor_id in [1, 2, 3]:
    if motor_id == 2:
        continue
    print(motor_id)
```

This prints `1` and `3`, but skips `2`.

## C++ Starter Concepts

### Include

C++ uses `#include` to bring in library features.

```cpp
#include <iostream>
```

This allows terminal printing with `std::cout`.

### Main Function

Every simple C++ program starts running from `main`.

```cpp
int main() {
    return 0;
}
```

### Variables

C++ variables need explicit types.

```cpp
int motor_id = 3;
double command = 0.5;
std::string motor_name = "hip_front_left";
bool enabled = true;
```

Python guesses types at runtime. C++ wants you to declare them.

### Function

```cpp
double clamp(double value, double minimum, double maximum) {
    if (value < minimum) {
        return minimum;
    }

    if (value > maximum) {
        return maximum;
    }

    return value;
}
```

This is the same idea as Python, but with types and braces.

### Class

A class is a blueprint for objects.

```cpp
class Motor {
public:
    int can_id;
    std::string name;
};
```

Then you can create an object:

```cpp
Motor hip_motor;
hip_motor.can_id = 3;
hip_motor.name = "hip_front_left";
```

## Python vs C++ Syntax

Python:

```python
def clamp(value, minimum, maximum):
    if value < minimum:
        return minimum
    return value
```

C++:

```cpp
double clamp(double value, double minimum, double maximum) {
    if (value < minimum) {
        return minimum;
    }
    return value;
}
```

The ideas are similar. The grammar is different.

Python uses indentation.
C++ uses braces `{}` and semicolons `;`.

## Robotics Translation

These beginner ideas become real robot software:

```text
variables       -> joint angles, motor IDs, sensor values
functions       -> clamp command, compute error, convert units
classes         -> Motor, Leg, RobotController
for loops       -> update every motor
while loops     -> keep the control loop running
break           -> stop searching when a motor is found
continue        -> skip disabled motors
C++             -> fast control, embedded systems, real-time logic
Python          -> learning, testing, plotting, simulation, tools
```
