// lesson02
#include <stdio.h>
#include <stdbool.h>

int main()
{
    // variable =  A reusable container for a value.
    // Behaves as if it were the value it contains.
    int motor_id = 1;
    float joint_target = 0.22;
    double moo = 21022.022;
    char happy_word = 'm';
    char name_moo[] = "ding_ding_cow";
    bool motor_online = 0;

    printf("the right leg shank joint uses %d as motor_id\n", motor_id);
    printf("The corresponding motor is rotating towards target %f now \n", joint_target);
    printf("moo is %.1lf\n", moo);
    printf("my goo goo ga ga is %c\n", happy_word);
    printf("hello little %s\n", name_moo);
    printf("%d\n", motor_online);

    if (motor_online)
    {
        printf("moomoo is online");
    }
    else
    {
        printf("you are offline");
    }

    return 0;
}