// Lesson15 For loop
#include <stdio.h>
#include <windows.h>

int main()
{
    for (int step = 0; step <= 10; step += 3)
    {
        Sleep(1000);
        printf("%d\n", step);
    }
    return 0;
}