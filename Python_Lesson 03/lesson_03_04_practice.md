# Lesson 03/04 Practice Problems

These short exercises match the ideas in `python_lesson_03_04_robot_ready.py`.

## 1. Tuple Practice

Create a foot position tuple:

```python
sleepy_foot = (0.0, 84.26, 410.0)
```

Unpack it into `x`, `y`, and `z`, then print each value.

Goal output:

```text
x is 0.0
y is 84.26
z is 410.0
```

## 2. Type Hint Practice

Write a function named `print_motor_name`.

It should take one argument:

```python
name: str
```

It should return nothing:

```python
-> None
```

Example:

```python
print_motor_name("front_right_hip")
```

Goal output:

```text
Motor name: front_right_hip
```

## 3. Return Type Practice

Write a function named `clamp_command`.

It should take:

```python
command: float
```

It should return:

```python
float
```

Rules:

- If `command` is greater than `1.0`, return `1.0`.
- If `command` is less than `-1.0`, return `-1.0`.
- Otherwise, return `command`.

Try it with:

```python
print(clamp_command(1.5))
print(clamp_command(-2.0))
print(clamp_command(0.25))
```

## 4. None Practice

Create this variable:

```python
active_target: float | None = None
```

Then write an `if` statement:

```python
if active_target is None:
    print("No target yet")
```

After that, set:

```python
active_target = 0.25
```

Then print:

```python
print("Active target:", active_target)
```

## 5. Dataclass Practice

Create this dataclass:

```python
from dataclasses import dataclass


@dataclass
class MotorConfig:
    name: str
    can_id: int
```

Then create:

```python
hip = MotorConfig("front_right_hip", 3)
```

Print:

```text
front_right_hip has CAN ID 3
```

## 6. F-String Formatting Practice

Create this variable:

```python
angle = 3.14159
```

Print it with two decimal places using:

```python
f"{angle:.2f}"
```

Goal output:

```text
Angle: 3.14
```

## 7. Robot Config Challenge

Create a dataclass named `JointConfig`.

It should store:

```python
joint_name: str
can_id: int
kp: float
kd: float
torque_limit: float
```

Then create:

```python
front_right_hip = JointConfig(
    joint_name="front_right_hip",
    can_id=3,
    kp=0.25,
    kd=0.015,
    torque_limit=1.30,
)
```

Print a summary using an f-string.

Goal output:

```text
front_right_hip ID=3 kp=0.25 kd=0.015 torque=1.30
```
