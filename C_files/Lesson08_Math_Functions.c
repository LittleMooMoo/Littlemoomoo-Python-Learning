// Lesson08 Math_Functions
#include <stdio.h>
#include <math.h>

float radius = 2.3;

int main()
{
    double pi = 3.141592653589793;
    double circumference = 2 * pi * radius;
    double area = pi * pow(radius, 2);
    double diagonal = sqrt(radius);
    double rounded_r = round(radius);
    printf("circumference is : %.3lf \n", circumference);
    printf("area is : %.3lf \n", area);
    printf("diagonal is : %.3lf \n", diagonal);
    printf("rounded radius is : %.3lf \n", rounded_r);
    float x = log(radius);
    printf("x : %.3lf \n", x);
    return 0;
}