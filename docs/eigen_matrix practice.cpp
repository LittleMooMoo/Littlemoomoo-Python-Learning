// eigen_matrix practice
#include <iostream>
#include <Eigen/Dense>
#include <cmath>

int main()
{
    Eigen::Matrix3d Moocobian;
    Moocobian << 1, 2, 3,
        3, 2, 1,
        5, 4, 2;

    Eigen::Vector3d Moopos;
    Moopos << 3, 2, 1;

    std::cout << Moocobian;
    std::cout << "\n"
              << Moopos;
    std::cout << "\n"
              << Moocobian * Moopos;
    return 0;
}