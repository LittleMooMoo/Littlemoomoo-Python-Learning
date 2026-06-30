// Lesson028_readfile
#include <stdio.h>
int main()
{
    FILE *pFile = fopen("input_example_file.txt", "r");
    char boxbox_moo_moo_boot[1024] = {0};

    if (pFile == NULL)
    {
        printf("can't open file\n");
        return 1;
    }

    while (fgets(boxbox_moo_moo_boot, sizeof(boxbox_moo_moo_boot), pFile) != NULL)
        ;
    {
        printf("%s\n", boxbox_moo_moo_boot);
    }
    fclose(pFile);
    return 0;
}