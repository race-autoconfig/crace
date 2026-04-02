#include <iostream>
#include <cstdlib>
#include <cmath>
#include <string>

double get_arg(int argc, char** argv, const std::string& name, double default_val = 0.0)
{
    for (int i = 1; i < argc - 1; ++i)
    {
        if (argv[i] == name)
            return std::atof(argv[i + 1]);
    }
    return default_val;
}

int get_int_arg(int argc, char** argv, const std::string& name, int default_val = 0)
{
    for (int i = 1; i < argc - 1; ++i)
    {
        if (argv[i] == name)
            return std::atoi(argv[i + 1]);
    }
    return default_val;
}

int main(int argc, char** argv)
{
    int instance = get_int_arg(argc, argv, "-i",
                   get_int_arg(argc, argv, "--instance", 1));

    int seed = get_int_arg(argc, argv, "--seed", 0);

    double x = get_arg(argc, argv, "-x", 0.0);
    double y = get_arg(argc, argv, "-y", 0.0);

    double a = 3.0 + instance;
    double b = -1.0 - instance;

    double value = (x - a)*(x - a) + (y - b)*(y - b);

    std::cout << value << std::endl;

    return 0;
}