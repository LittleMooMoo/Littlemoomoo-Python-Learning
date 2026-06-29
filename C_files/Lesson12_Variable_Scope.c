// Lesson12 Variable Scope
#include <stdio.h>

int add(int x, int y)
{
    int result = x + y;
    return result;
}

int main()
{
    int result = add(3, 4);
    printf("%d", result);
}
