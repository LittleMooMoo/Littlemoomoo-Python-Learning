// Lesson17 nested loops
#include <stdio.h>

int main()
{
    /*for (int j = 1; j <= 4; j++)
    {
        for (int i = 1; i < 10; i++)
        {
            printf(" %d ", i);
        }
        printf("\n");
    }*/

    /*for (int i = 1; i <= 9; i++)
    {
        for (int j = 1; j <= 9; j++)
        {
            printf(" %3d ", i * j);
        }
        printf("\n");
    }*/
    int number_of_rows;
    int number_of_columns;
    char symbol = '\0';

    printf("please enter number of rows: ");
    scanf("%d", &number_of_rows);

    printf("\nplease enter number of columns: ");
    scanf("%d", &number_of_columns);

    printf("\nplease enter symbols: ");
    scanf(" %c", &symbol);

    for (int j = 0; j < number_of_rows; j++)
    {
        for (int i = 0; i < number_of_columns; i++)
        {

            printf(" %c ", symbol);
        }
        printf("\n");
    }
    return 0;
}