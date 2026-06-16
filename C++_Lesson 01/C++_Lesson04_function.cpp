#include <iostream>
#include <cmath>
#include "rad2deg.h"

double L_H = 84.26;
double L_T = 270.0;
double L_S = 270.0;
double pi = 3.141592653589793;

struct JointAngles
{
    double theta_h;
    double theta_t;
    double theta_s;
};

JointAngles leg_ik(double x, double y, double z)
{
    double z_1 = std::sqrt(y * y + z * z - L_H * L_H);
    double l_1 = std::sqrt(x * x + z_1 * z_1);
    double a_1 = std::acos((L_S * L_S + L_T * L_T - l_1 * l_1) / (2 * L_S * L_T));
    double a_2 = std::atan2(x, z_1);
    double b_1 = std::acos((L_T * L_T + l_1 * l_1 - L_S * L_S) / (2 * L_T * l_1));
    double b_2 = std::atan2(y, z);
    double b_3 = std::atan2(z_1, L_H);

    JointAngles angles;
    angles.theta_h = pi / 2 - b_2 - b_3;
    angles.theta_t = b_1 - a_2;
    angles.theta_s = pi - a_1;
    return angles;
}

int main()
{
    double x;
    double y;
    double z;
    std::cout << "please enter cartesian coodrinate foot end position as x y z:\n";
    std::cin >> x >> y >> z;

    JointAngles angles = leg_ik(x, y, z);
    std::cout << "theta_h :" << angles.theta_h << "\n";
    std::cout << "theta_t :" << angles.theta_t << "\n";
    std::cout << "theta_s :" << angles.theta_s << "\n";

    double degrees_theta_h = radians_to_degrees(angles.theta_h);
    double degrees_theta_t = radians_to_degrees(angles.theta_t);
    double degrees_theta_s = radians_to_degrees(angles.theta_s);
    std::cout << "theta_h in degrees:" << degrees_theta_h << "\n";
    std::cout << "theta_t in degrees:" << degrees_theta_t << "\n";
    std::cout << "theta_s in degrees:" << degrees_theta_s << "\n";
}