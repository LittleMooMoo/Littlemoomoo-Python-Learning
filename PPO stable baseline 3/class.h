#include <string>
#include <array>
#include <iostream>

class MotorController
{
public:
    MotorController(int id, std::string bus, int dir, float limit);
    int getid();

private:
    int id;
    std::string bus;
    int dir;
    float limit;
};