// Practice Eigen / Dense
#include <iostream>
#include <Eigen/Dense>

int main()
{
    Eigen::Matrix2d matrix;
    matrix << 1.0, 2.0,
        3.0, 4.0;

    Eigen::Vector2d vector;
    vector << 10.0, 20.0;

    Eigen::Vector2d result = matrix * vector;

    std::cout << "matrix:\n"
              << matrix << "\n\n";
    std::cout << "vector:\n"
              << vector << "\n\n";
    std::cout << "matrix * vector:\n"
              << result << "\n";

    return 0;
}
