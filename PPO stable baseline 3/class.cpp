#include "class.h"

MotorController::MotorController(int id, std::string bus, int dir, float limit)
    : id{id},
      bus{bus},
      dir{dir},
      limit{limit}
{
}

int MotorController::getid()
{
    return {id};
}

int main()
{
    MotorController motor(1, "can1", 2, 1);
    int idd;
    idd = motor.getid();
    std::cout << "configs: " << idd;
    if (a == b)
    {
        a = 1
    }
}