#include <iostream>
#include "Core/model.hpp"

int main(int argc, const char **argv)
{
    std::cout << argv[0] << std::endl;
    double **inp = (double **)malloc(sizeof(double **));
    double **out = (double **)malloc(sizeof(double **));

    int j = 1;
    for(int i = 0; i < 3; ++i)
    {
        inp[i] = (double *)malloc(sizeof(double) * 3);
        inp[i][0] = 1; inp[i][1] = 2 + 1; inp[i][2] = 3;
        j += 1;
    }j = 1;

    for(int i = 0; i < 3; ++i)
    {
        out[i] = (double *)malloc(sizeof(double) * 3);
        out[i][0] = i + j; out[i][1] = i + j + 1; out[i][2] = i + j + 2;
        j += 1;
    }

    unsigned inp_format[2] = {3, 3};

    unsigned epochs = 3;

    Dense<double> *model = new Dense<double>(true, epochs, inp_format, true, 0.01, "MeanSqrErr");
    model->add(3, "relu");
    model->add(2, "leaky_relu");
    model->add(3, "relu");

    model->initialize_network_input(inp);
    model->initialize_network_output(out);

    model->initialize_global_variables();
    std::cout << "\n\n\n\n\n\n";

    model->train();

    return 0;
}
