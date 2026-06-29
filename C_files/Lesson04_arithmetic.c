// Lesson04 arithmetic
#include <stdio.h>

int main()
{
    float x = 2;
    float y = 3;
    float z = 4;
    int moo = 2;
    int ga = 3;

    float sum_all = x + y + z;
    float subtract_all = x - y - z;
    float division = x / y;
    // int mewo = moo / ga ;
    int ding = moo * moo;

    int remainder_meow = ga % moo;
    // x--;

    x = x + 2;
    x += 2;
    x /= 2.5;

    printf("%.1f\n", sum_all);
    printf("%.1f\n", subtract_all);
    printf("%.4f\n", division);
    printf("%d\n", remainder_meow);
    printf("%d\n", ding);
    printf("%f\n", x);
    return 0;
}