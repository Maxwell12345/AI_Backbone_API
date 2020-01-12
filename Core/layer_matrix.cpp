#include <iostream>
#include <cmath>
#include <vector>
#include "extern_files.hpp"

template<class T>
class Layer
{
public:
    //Used for input and hidden layers
    Layer(int nRow, int nCol, std::string _activation_func_, bool weight_range);
    //Used for Output layers
    Layer(int nRow, std::string _activation_func_);

    ~Layer(void);

    Eigen::Matrix<T, -1, 1> format_input(T *inp);
    Eigen::Matrix<T, -1, 1> format_input(std::vector<T> inp);
    static Eigen::Matrix<T, -1, -1> format_variable_mat(T ** mat, unsigned num_rows, unsigned num_cols);
    
    inline void init_Mat2D(void);
    inline void set_Mat2D(Eigen::Matrix<T, -1, -1> updated_mat);
    inline T get_variable_val(unsigned idY, unsigned idX);
    inline Eigen::Matrix<T, -1, -1> get_variable_mat(void);
    
    inline void set_NeuronArr1D(Eigen::Matrix<T, -1, 1>);
    inline T get_neuron_val(unsigned idY);
    inline Eigen::Matrix<T, -1, 1> get_neuron_mat(void);

    Eigen::Matrix<T, -1, 1> __weight_to_neuron_matMul__(void);
    inline void feed_forward(Layer<T> *&next);
    
    void toString(void);

    //Prints final output values
    void toString(int);

private:
    Eigen::Matrix<T, -1, -1> Mat2D;
    Eigen::Matrix<T, -1, 1> NeuronArr1D;
    std::string _activation_func_;
    bool weight_range;

    //Impliment bias array later
    // bias array

protected:
    int num_rows;
    int num_cols;
};
