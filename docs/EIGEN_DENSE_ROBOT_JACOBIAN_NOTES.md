# Eigen/Dense Notes For Robot Dog Jacobians

This note summarizes the Eigen syntax and objects you should know first for
matrix calculations in C++ robotics code.

Your local Eigen folder is:

```text
C&C++_lib/Eigen
```

Use this include in C++:

```cpp
#include <Eigen/Dense>
```

Compile from the repo root:

```powershell
g++ -std=c++17 -Wall -Wextra -I"C&C++_lib\Eigen" "C++_File\practice_eigen_dense.cpp" -o "C++_File\practice_eigen_dense.exe"
```

If you compile from inside `C++_File`, use:

```powershell
g++ -std=c++17 -Wall -Wextra -I"..\C&C++_lib\Eigen" practice_eigen_dense.cpp -o practice_eigen_dense.exe
```

## The Most Important Eigen Types

Fixed-size vectors:

```cpp
Eigen::Vector2d v2;  // 2 x 1 vector of double
Eigen::Vector3d v3;  // 3 x 1 vector of double
Eigen::Vector4d v4;  // 4 x 1 vector of double
```

Fixed-size matrices:

```cpp
Eigen::Matrix2d A2;  // 2 x 2 matrix of double
Eigen::Matrix3d A3;  // 3 x 3 matrix of double
Eigen::Matrix4d A4;  // 4 x 4 matrix of double
```

Dynamic-size vectors and matrices:

```cpp
Eigen::VectorXd q(12);       // 12 x 1 vector, size chosen at runtime
Eigen::MatrixXd J(3, 12);    // 3 x 12 matrix, size chosen at runtime
```

Custom fixed-size matrix:

```cpp
Eigen::Matrix<double, 3, 2> J;   // 3 rows, 2 columns
Eigen::Matrix<double, 12, 1> q;  // 12 x 1 vector
```

For one 3-DOF robot leg, you will often use:

```cpp
Eigen::Vector3d q_leg;       // hip, thigh, shank joint angles
Eigen::Vector3d foot_pos;    // x, y, z foot position
Eigen::Matrix3d J_leg;       // maps joint velocity to foot velocity
```

## Creating Matrices And Vectors

Zero:

```cpp
Eigen::Vector3d v = Eigen::Vector3d::Zero();
Eigen::Matrix3d J = Eigen::Matrix3d::Zero();
```

Identity:

```cpp
Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
```

Constant:

```cpp
Eigen::Vector3d gains = Eigen::Vector3d::Constant(0.2);
```

Manual values:

```cpp
Eigen::Vector3d q;
q << 0.1, -0.4, 0.8;
```

Matrix manual values:

```cpp
Eigen::Matrix3d J;
J << 1.0, 0.0, 0.2,
     0.0, 1.0, 0.1,
     0.0, 0.0, 1.0;
```

Important: Eigen fills using row-major-looking comma syntax, even though its
internal storage is column-major by default. For beginner use, just write the
matrix as it visually appears.

## Accessing Elements

Vector:

```cpp
double hip = q(0);
double thigh = q(1);
double shank = q(2);

q(0) = 0.2;
```

Matrix:

```cpp
double value = J(0, 2);  // row 0, column 2
J(1, 0) = 3.5;
```

Rows and columns:

```cpp
int rows = J.rows();
int cols = J.cols();
```

## Basic Matrix Math

Addition and subtraction:

```cpp
Eigen::Vector3d error = target - measured;
```

Scalar multiplication:

```cpp
Eigen::Vector3d command = 0.5 * error;
```

Matrix-vector multiplication:

```cpp
Eigen::Vector3d foot_velocity = J_leg * joint_velocity;
```

Matrix-matrix multiplication:

```cpp
Eigen::Matrix3d C = A * B;
```

Transpose:

```cpp
Eigen::Matrix3d JT = J.transpose();
```

Inverse:

```cpp
Eigen::Matrix3d J_inv = J.inverse();
```

For real control code, avoid using `.inverse()` unless you are learning or the
matrix is tiny and safe. Prefer solving equations.

## Vector Operations

Norm:

```cpp
double length = error.norm();
```

Normalize:

```cpp
Eigen::Vector3d direction = error.normalized();
```

Dot product:

```cpp
double work_like_value = force.dot(velocity);
```

Cross product:

```cpp
Eigen::Vector3d torque = lever_arm.cross(force);
```

For robotics intuition:

```text
dot product   -> projection / power-like calculations
cross product -> torque from force and lever arm
```

## Blocks, Rows, Columns, And Segments

Get one row:

```cpp
Eigen::RowVector3d row0 = J.row(0);
```

Get one column:

```cpp
Eigen::Vector3d col1 = J.col(1);
```

Get part of a vector:

```cpp
Eigen::Vector3d front_right_q = q_all.segment<3>(0);
Eigen::Vector3d front_left_q  = q_all.segment<3>(3);
```

Write part of a vector:

```cpp
q_all.segment<3>(0) = front_right_q;
```

Get a block from a matrix:

```cpp
Eigen::Matrix3d J_front_right = J_all.block<3, 3>(0, 0);
```

Dynamic block:

```cpp
Eigen::MatrixXd block = J_all.block(row_start, col_start, row_count, col_count);
```

For a quadruped, this matters because your full robot vector may be:

```text
q_all = [FR_hip, FR_thigh, FR_shank,
         FL_hip, FL_thigh, FL_shank,
         RR_hip, RR_thigh, RR_shank,
         RL_hip, RL_thigh, RL_shank]
```

Each leg can use a 3-value segment.

## Solving Linear Systems

Instead of:

```cpp
Eigen::Vector3d x = A.inverse() * b;
```

prefer:

```cpp
Eigen::Vector3d x = A.colPivHouseholderQr().solve(b);
```

Common solvers:

```cpp
A.colPivHouseholderQr().solve(b);  // general matrix, good beginner default
A.ldlt().solve(b);                 // symmetric matrix
A.llt().solve(b);                  // symmetric positive definite matrix
```

For robotics:

```text
A x = b
```

means:

```text
Find x such that A times x equals b.
```

In Jacobian IK:

```text
J * dq = foot_error
```

means:

```text
Find joint change dq that reduces the foot error.
```

## Jacobian Meaning

For one leg:

```text
foot_velocity = J_leg(q) * joint_velocity
```

In Eigen:

```cpp
Eigen::Vector3d foot_velocity = J_leg * q_dot;
```

Dimensions:

```text
J_leg         = 3 x 3
q_dot         = 3 x 1
foot_velocity = 3 x 1
```

For a 3-DOF leg:

```text
columns of J = how the foot moves when each joint moves
```

Column 0: foot motion caused by hip motion.

Column 1: foot motion caused by thigh motion.

Column 2: foot motion caused by shank motion.

## Basic Jacobian IK

If the foot is at the wrong place:

```cpp
Eigen::Vector3d foot_error = target_foot_pos - measured_foot_pos;
```

You want a joint change:

```cpp
Eigen::Vector3d dq = J_leg.inverse() * foot_error;
```

This is the simple learning version.

But in real robot code, use a solver:

```cpp
Eigen::Vector3d dq = J_leg.colPivHouseholderQr().solve(foot_error);
```

Then update joint command:

```cpp
q_command = q_current + dq;
```

## Damped Least-Squares IK

Damped least-squares is safer near singular positions.

Formula:

```text
dq = J^T * (J * J^T + lambda^2 * I)^-1 * foot_error
```

Eigen version:

```cpp
Eigen::Vector3d damped_ik_step(
    const Eigen::Matrix3d& J,
    const Eigen::Vector3d& foot_error,
    double lambda)
{
    Eigen::Matrix3d damping =
        lambda * lambda * Eigen::Matrix3d::Identity();

    Eigen::Matrix3d A = J * J.transpose() + damping;

    Eigen::Vector3d dq =
        J.transpose() * A.ldlt().solve(foot_error);

    return dq;
}
```

Use it like:

```cpp
Eigen::Vector3d dq = damped_ik_step(J_leg, foot_error, 0.01);
Eigen::Vector3d q_next = q_current + dq;
```

What `lambda` does:

```text
small lambda -> faster/more aggressive
large lambda -> safer/slower
```

Start with a small value like:

```cpp
double lambda = 0.01;
```

Then tune carefully.

## Dynamic-Size Damped IK

For a leg or robot where dimensions may change:

```cpp
Eigen::VectorXd damped_ik_step_dynamic(
    const Eigen::MatrixXd& J,
    const Eigen::VectorXd& task_error,
    double lambda)
{
    Eigen::MatrixXd I =
        Eigen::MatrixXd::Identity(J.rows(), J.rows());

    Eigen::MatrixXd A =
        J * J.transpose() + lambda * lambda * I;

    Eigen::VectorXd dq =
        J.transpose() * A.ldlt().solve(task_error);

    return dq;
}
```

Dimensions:

```text
J          = task_dim x joint_dim
task_error = task_dim x 1
dq         = joint_dim x 1
```

Example:

```text
one foot xyz task:
J = 3 x 3

four feet xyz task:
J = 12 x 12

body + feet task:
J might be 18 x 12 or another shape
```

## Force To Joint Torque

The Jacobian also maps foot force to joint torque:

```text
joint_torque = J^T * foot_force
```

Eigen:

```cpp
Eigen::Vector3d joint_torque =
    J_leg.transpose() * foot_force;
```

This is important for:

```text
stance control
virtual model control
MPC force control
whole-body control
```

If MPC gives you a foot contact force:

```cpp
Eigen::Vector3d foot_force;
```

you can convert it to joint torque with:

```cpp
Eigen::Vector3d tau = J_leg.transpose() * foot_force;
```

## Function Parameters With Eigen

Pass Eigen objects by `const reference` when you only read them:

```cpp
double foot_error_norm(const Eigen::Vector3d& error)
{
    return error.norm();
}
```

Return Eigen objects normally:

```cpp
Eigen::Vector3d calculate_error(
    const Eigen::Vector3d& target,
    const Eigen::Vector3d& measured)
{
    return target - measured;
}
```

Modify an output object by reference:

```cpp
void fill_jacobian(Eigen::Matrix3d& J)
{
    J.setZero();
    J(0, 0) = 1.0;
    J(1, 1) = 1.0;
    J(2, 2) = 1.0;
}
```

Good beginner pattern:

```cpp
const Eigen::Vector3d& input_that_I_read
Eigen::Vector3d& output_that_I_modify
```

## Useful Setters

Set all zeros:

```cpp
J.setZero();
q.setZero();
```

Set identity:

```cpp
J.setIdentity();
```

Set constant:

```cpp
q.setConstant(0.1);
```

Resize dynamic matrices:

```cpp
Eigen::MatrixXd J;
J.resize(3, 12);
J.setZero();
```

## Printing Eigen Objects

```cpp
std::cout << "J:\n" << J << "\n";
std::cout << "q:\n" << q.transpose() << "\n";
```

Why `q.transpose()`?

```text
Vector3d normally prints vertically.
transpose() prints it as one horizontal row.
```

## Common Beginner Errors

Wrong include:

```cpp
#include < Eigen / Dense>  // wrong
```

Correct include:

```cpp
#include <Eigen/Dense>
```

Missing include path:

```text
fatal error: Eigen/Dense: No such file or directory
```

Fix:

```powershell
g++ -I"C&C++_lib\Eigen" ...
```

Dimension mismatch:

```cpp
Eigen::Vector3d y = Eigen::Matrix2d::Identity() * Eigen::Vector3d::Ones();
```

Wrong because:

```text
2 x 2 times 3 x 1 does not fit.
```

Correct dimensions:

```cpp
Eigen::Vector2d y =
    Eigen::Matrix2d::Identity() * Eigen::Vector2d::Ones();
```

Using inverse too much:

```cpp
x = A.inverse() * b;  // works sometimes, but not preferred
```

Better:

```cpp
x = A.colPivHouseholderQr().solve(b);
```

## Robot Dog Learning Checklist

Learn these in order:

1. `Vector3d` for foot position and joint angles.
2. `Matrix3d` for one leg Jacobian.
3. Matrix-vector multiplication: `foot_velocity = J * q_dot`.
4. Solving: `dq = J.colPivHouseholderQr().solve(foot_error)`.
5. Damped least-squares IK.
6. `J.transpose() * foot_force` for stance/contact force to torque.
7. `VectorXd` and `MatrixXd` for all 12 joints.
8. `.segment<3>()` to split full robot vectors into legs.
9. `.block<3, 3>()` to split full robot Jacobians into leg Jacobians.

## Tiny Complete Example

```cpp
#include <Eigen/Dense>
#include <iostream>

Eigen::Vector3d damped_ik_step(
    const Eigen::Matrix3d& J,
    const Eigen::Vector3d& foot_error,
    double lambda)
{
    Eigen::Matrix3d A =
        J * J.transpose()
        + lambda * lambda * Eigen::Matrix3d::Identity();

    return J.transpose() * A.ldlt().solve(foot_error);
}

int main()
{
    Eigen::Vector3d q_current;
    q_current << 0.1, -0.5, 0.8;

    Eigen::Vector3d foot_error;
    foot_error << 0.02, -0.01, 0.03;

    Eigen::Matrix3d J;
    J << 0.0, 0.10, 0.05,
         0.2, 0.00, 0.00,
         0.0, 0.30, 0.15;

    Eigen::Vector3d dq = damped_ik_step(J, foot_error, 0.01);
    Eigen::Vector3d q_next = q_current + dq;

    std::cout << "dq: " << dq.transpose() << "\n";
    std::cout << "q_next: " << q_next.transpose() << "\n";

    return 0;
}
```

The robotics meaning:

```text
foot_error tells you where the foot should move.
J tells you how joint changes move the foot.
damped_ik_step calculates a safe joint update.
q_next becomes the next joint target.
```
