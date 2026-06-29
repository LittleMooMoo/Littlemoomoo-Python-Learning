// Lesson 03 format specifier
#include <stdio.h>
#include <stdbool.h>

int main()
{
    /*
    int moo_moo_value = 1;
    float moo = 2.5;
    double pi = 3.141592653589793;
    char goo = 'm';

    char name_moo[] = "goo goo ga ga";

    printf("%d/n", moo_moo_value);
    printf("%f\n", moo);
    printf("%lf\n", pi);
    printf("%c\n", goo);
    printf("%s\n", name_moo);
    */

    /*
    int num1 = 1;
    int num2 = 2;
    int num3 = 3;
    printf("%01d\n", num1);
    printf("%03d\n", num2);
    printf("%02d\n", num3);
    */

    float command_target = 2.33;
    float measued_target = 0.22;
    float measued_velocity = 1.00;
    printf("%+4.2f\n", command_target);
    printf("%-4.2f\n", measued_target);
    printf("%+4.2f\n", measued_velocity);

    return 0;
}