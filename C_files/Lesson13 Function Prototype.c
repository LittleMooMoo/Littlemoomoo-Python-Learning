// Lesson13 Function Prototype
#include <stdio.h>
#include <stdbool.h>

void googoogaga(char moo_moo_name[], char favorite_grass[]);

int main()
{
    char moo_moo_name[] = "little moo moo";
    char moo_moo_grass[] = "googoogaga grass";
    googoogaga(moo_moo_name, moo_moo_grass);

    return 0;
}

void googoogaga(char moo_moo_name[], char favorite_grass[])
{
    printf("Hello %s\n ", moo_moo_name);
    printf("Your favorite grass is : %s", favorite_grass);
}
