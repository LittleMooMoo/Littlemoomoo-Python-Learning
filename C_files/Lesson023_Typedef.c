// Lesson023_Typedef
#include <stdio.h>

typedef int Number;
typedef char initials[3];

int main()
{
    // typedef = reserved keyword that gives an existing datatype a nickname;
    Number x = 3;
    Number y = 4;
    Number z = x + y;
    printf("%d", z);

    initials moo1 = "goo";
    printf("%s", moo1);

    return 0;
}