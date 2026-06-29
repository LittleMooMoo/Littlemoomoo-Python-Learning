// Lesson20_Array_and_userinput
#include <stdio.h>

int main()
{
    int matrixA[5] = {0};

    printf("enter a starting elements: ");
    scanf("%d", &matrixA[0]);

    for (int i = 0; i < 5; i++)
    {
        printf("enter a starting elements: ");
        scanf("%d", &matrixA[i]);
    }
    for (int i = 0; i < 5; i++)
    {
        printf("%d", matrixA[i]);
    }

    return 0;
}