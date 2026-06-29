// Lesson11 Logic Operators
#include <stdio.h>
#include <stdbool.h>

int main()
{
    //&& = AND
    //|| = OR
    // ! = NOT
    /*float motor_target = 45;
    if (motor_target > 0 || motor_target < 40)
    {
        printf("%.2f", motor_target);
    }
    */
    bool motor_is_online = false;
    if (!motor_is_online)
    {
        printf("motor is online");
    }
    return 0;
}