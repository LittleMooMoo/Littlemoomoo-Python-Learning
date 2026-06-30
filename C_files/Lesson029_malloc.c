// Lesson029_malloc
#include <stdio.h>
#include <stdlib.h>

int main()
{
    // malloc() =
    // A function in C that dynamically
    // allocates a specified number of bytes in memory;
    int num = 0;
    printf("enter the number of gpa: ");
    scanf("%d", &num);

    char *gpa = malloc(num * sizeof(char));

    if (gpa == NULL)
    {
        printf("memory allocation failed\n");
        return 1;
    }

    for (int i = 0; i < num; i++)
    {
        printf("enter gpa number #%d ", i + 1);
        scanf(" %c", &gpa[i]);
    }
    for (int i = 0; i < num; i++)
    {
        printf("%c", gpa[i]);
    }
    free(gpa);  // returning the "rented space back to OS"
    gpa = NULL; // avoid dangling pointers

    return 0;
}