// Lesson 22_ternary_operator
#include <stdio.h>
#include <stdbool.h>

/*int main()
{
    int x = 5;
    int y = 6;
    int max = (x > y) ? x : y; //return x if x >y, return y if x < y;

    printf("%d", max);
    return 0;
}*/

/*int main()
{
    bool isOline = false;

    printf("%s", (isOline) ? "online" : "offline");
    return 0;
}*/

/*int main()
{
    int x;
    printf("please enter an integer: ");
    scanf("%d", &x);
    printf(" %d is %s", x, (x % 2 == 0) ? "even" : "odd");
    return 0;
}*/

/*int main()
{
    int age = 21;
    printf("%s", (age > 18) ? "adult" : "child");
}*/

int main()
{
    int time = 14;
    int min = 3;
    char *moo = (time < 12) ? "AM" : "PM";

    printf("%02d:%02d %s", time, min, moo);
    return 0;
}