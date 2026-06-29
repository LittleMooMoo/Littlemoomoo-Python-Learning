// Lesson16 break and continue
#include <stdio.h>
#include <Windows.h>

int main()
{
    for (float i = 0.0; i <= 10; i += 0.5)
    {
        if (i == 5.00)
        {
            Sleep(500);
            printf("%.2f\n", i);
            break;
        }
        else
        {
            Sleep(500);
            i = i + 2;
            printf("%.2f\n", i);
        }
    }
    return 0;
}