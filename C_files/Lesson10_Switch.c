// Lesson10 Switch
#include <stdio.h>
int motor_id = 2;

int main()
{
    switch (motor_id)
    {
    case 1:
        printf("it is right front shank");
        break;
    case 2:
        printf("it is right front thigh");
        break;

    default:
        break;
    }
}