// Lesson027_write_files
#include <stdio.h>
int main()
{

    FILE *pFILE = fopen("C:\\Users\\jiaxi\\OneDrive\\Desktop\\moomoo_file.txt", "w");
    char tetx[] = "moo moo little ding ding cow moo";

    if (pFILE == NULL)
    {
        printf("Error opening file\n");
        return 1;
    }
    fprintf(pFILE, "%s", tetx);
    printf("file was written successfully");
    fclose(pFILE);

    return 0;
}