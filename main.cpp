#include <iostream>
#include "Core/model.hpp"

void setData(std::vector<std::vector<double> > &inp, std::vector<std::vector<double> > &out)
{
    std::vector<double> d;
    d.push_back(0);
    d.push_back(0);
    d.push_back(0);
    d.push_back(0); //0
    inp.push_back(d);
    d.clear();

    d.push_back(0);
    d.push_back(0);
    d.push_back(0);
    d.push_back(1); //0
    inp.push_back(d);
    d.clear();

    d.push_back(0);
    d.push_back(0);
    d.push_back(1);
    d.push_back(1); //1
    inp.push_back(d);
    d.clear();

    d.push_back(0);
    d.push_back(1);
    d.push_back(1);
    d.push_back(1); //0
    inp.push_back(d);
    d.clear();

    d.push_back(1);
    d.push_back(1);
    d.push_back(1);
    d.push_back(1); //0
    inp.push_back(d);
    d.clear();

    d.push_back(1);
    d.push_back(0);
    d.push_back(0);
    d.push_back(0); //0
    inp.push_back(d);
    d.clear();

    d.push_back(1);
    d.push_back(0);
    d.push_back(0);
    d.push_back(1); //1
    inp.push_back(d);
    d.clear();

    d.push_back(1);
    d.push_back(0);
    d.push_back(1);
    d.push_back(1); //0
    inp.push_back(d);
    d.clear();

    d.push_back(1);
    d.push_back(1);
    d.push_back(0);
    d.push_back(0); //1
    inp.push_back(d);
    d.clear();

    d.push_back(1);
    d.push_back(1);
    d.push_back(0);
    d.push_back(1); //0
    inp.push_back(d);
    d.clear();

    d.push_back(1);
    d.push_back(1);
    d.push_back(1);
    d.push_back(0); //0
    inp.push_back(d);
    d.clear();

    d.push_back(0);
    d.push_back(1);
    d.push_back(0);
    d.push_back(1); //1
    inp.push_back(d);
    d.clear();

    d.push_back(1);
    d.push_back(0);
    d.push_back(1);
    d.push_back(0); //1
    inp.push_back(d);
    d.clear();

    d.push_back(0);
    d.push_back(1);
    d.push_back(1);
    d.push_back(0); //1
    inp.push_back(d);
    d.clear();
    
    // 0 0 1 0 0 0 1 0 1 0 0 1 1 1

    d.push_back(0);
    out.push_back(d);
    d.clear();
    d.push_back(0);
    out.push_back(d);
    d.clear();
    d.push_back(1);
    out.push_back(d);
    d.clear();
    d.push_back(0);
    out.push_back(d);
    d.clear();
    d.push_back(0);
    out.push_back(d);
    d.clear();
    d.push_back(0);
    out.push_back(d);
    d.clear();
    d.push_back(1);
    out.push_back(d);
    d.clear();
    d.push_back(0);
    out.push_back(d);
    d.clear();
    d.push_back(1);
    out.push_back(d);
    d.clear();
    d.push_back(0);
    out.push_back(d);
    d.clear();
    d.push_back(0);
    out.push_back(d);
    d.clear();
    d.push_back(1);
    out.push_back(d);
    d.clear();
    d.push_back(1);
    out.push_back(d);
    d.clear();
    d.push_back(1);
    out.push_back(d);
    d.clear();
}

int main(int argc, const char **argv)
{
    std::cout << argv[0] << std::endl;
    std::vector<std::vector<double> > inp;
    std::vector<std::vector<double> > out;
    setData(inp, out); // xor gate io

    unsigned int* inp_format = NULL; inp_format = (unsigned int *)calloc(2, sizeof(unsigned int));
    inp_format[0] = 14; inp_format[1] = 4;

    unsigned epochs = 200;

    Dense<double> *model = new Dense<double>(true, epochs, true, 0.01, "MeanSqrErr");
    model->set_input_shape(inp_format);
    model->add(4, "linear");
    model->bias(1);
    model->add(3, "linear");
    //model->bias(1);
    // model->add(4, "tanh");
    model->add(1, "linear"); // Need to work of backprop algorithms for cat_crossentropy and MeanSqrErr etc.

    model->initialize_network_input(inp);
    model->initialize_network_output(out);

    model->initialize_global_variables();

    model->train();
    return 0;
}
