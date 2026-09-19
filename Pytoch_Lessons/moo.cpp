#include <iostream>

float coweatgrass(bool happy)
{
    float happy_level;
    if (happy == true)
    {
        happy_level = 100;
    }
    return happy_level;
}

int main()
{
    float level = coweatgrass(true);
    std::cout << "happy level: " << level;
    return 0;
}