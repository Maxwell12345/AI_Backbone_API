//
//  layer_matrix.hpp
//  AI_Backbone
//
//  Created by maxwell on 12/10/2019.
//  Copyright © 2019 organized-organization. All rights reserved.
//

#ifndef layer_matrix_hpp
#define layer_matrix_hpp
#include <iostream>
#include <ctime>
#include <random>
#include "extern_files.hpp"
#include "activation_data.hpp"
#include "layer_matrix.cpp"

template<class T> inline T
__initialize_rand_weight__(bool weight_range) 
// weight_range is true --> negative bound weight values are possible
{
    srand(unsigned(time(NULL)));
    int isNeg = arc4random() % 10000;
    T rVal = ((T)(arc4random() % 10000) / 10000);
    if(weight_range)
    {
        if(isNeg % 2 == 0)
        {
            
            return (T)rVal;
        }
        else
        {
            return (T)rVal * -1;
        }
    }
    else
    {
        return (T)rVal;
    }
}

//Returns the multiplied and activated sums of the neurons and weights in the network
template<class T> Eigen::Matrix<T, -1, 1>
Layer<T> :: __weight_to_neuron_matMul__()
{
    // Initializes the matrix to be returned
    Eigen::Matrix<T, -1, 1> return_disposable;
    T var_disposable = 0;
    return_disposable.resize(this->num_cols, 1);

    uint32_t i = 0, j = 0;

    while(i < this->num_cols)
    {
        while(j < this->num_rows)
        {
            var_disposable += this->get_neuron_val(j) * this->get_variable_val(j, i);
            j++;
        }
        return_disposable.row(i).col(0) << __ret_activated_val__(var_disposable, this->_activation_func_);

        //reset var_dispoasable to itial state
        var_disposable = 0;
        j = 0;
        i++;
    }

    return return_disposable;
}

template<class T>
Layer<T> :: Layer(int nRow, int nCol, std::string _activation_func_, bool weight_range)
{
    this->num_rows = nRow;
    this->num_cols = nCol;
    this->weight_range = weight_range;
    this->_activation_func_ = _activation_func_;
    this->Mat2D.resize(nRow, nCol);
    this->NeuronArr1D.resize(nRow, 1);
}

template<class T>
Layer<T> :: Layer(int nRow, std::string _activation_func_)
{
    this->num_rows = nRow;
    this->num_cols = 1;
    this->_activation_func_ = _activation_func_;
    this->Mat2D.resize(0, 0);
    this->NeuronArr1D.resize(nRow, 1);
}

template<class T> inline void
Layer<T> :: set_Mat2D(Eigen::Matrix<T, -1, -1> updated_mat)
{
    updated_mat.resize(this->num_rows, this->num_cols);
    uint32_t i = 0, j = 0;
    while(i < this->num_rows)
    {
        while(j < this->num_cols)
        {
            this->Mat2D.row(i).col(j) << updated_mat(i, j);
            j++;
        }
        j = 0;
        i++;
    }
}

template<typename T> inline void
Layer<T> :: init_Mat2D()
{
    uint32_t i = 0, j = 0;

    while(i < this->num_rows)
    {
        while(j < this->num_cols)
        {
            this->Mat2D.row(i).col(j) << __initialize_rand_weight__<T>(this->weight_range);
            j++;
        }
        i++;
        j = 0;
    }
}

template<class T> inline T
Layer<T> :: get_variable_val(unsigned idY, unsigned idX)
{
    return this->Mat2D(idY, idX);
}

template<class T> inline Eigen::Matrix<T, -1, -1>
Layer<T> :: get_variable_mat()
{
    return this->Mat2D;
}

template<class T> inline void
Layer<T> :: set_NeuronArr1D(Eigen::Matrix<T, -1, 1> updated_inp)
{
    updated_inp.resize(this->num_rows, 1);
    this->NeuronArr1D << updated_inp;
}

template<class T> inline T
Layer<T> :: get_neuron_val(unsigned idY)
{
    return this->NeuronArr1D(idY, 0);
}

template<class T> inline Eigen::Matrix<T, -1, 1>
Layer<T> :: get_neuron_mat()
{
    return this->NeuronArr1D;
}

template<class T> Eigen::Matrix<T, -1, 1>
Layer<T> :: format_input(T *inp)
{
    Eigen::Matrix<T, -1, 1> return_disposable;
    return_disposable.resize(this->num_rows, 1);
    uint32_t i = 0;

    while(i < this->num_rows)
    {
        return_disposable.row(i).col(0) << inp[i];
        i++;
    }

    return return_disposable;
}

template<class T> Eigen::Matrix<T, -1, 1>
Layer<T> :: format_input(std::vector<T> inp)
{
    Eigen::Matrix<T, -1, 1> return_disposable;
    return_disposable.resize(this->num_rows, 1);
    uint32_t i = 0;

    while(i < this->num_rows)
    {
        return_disposable.row(i).col(0) << inp[i];
        i++;
    }
    return return_disposable;
} 

template<class T> Eigen::Matrix<T, -1, -1>
Layer<T> :: format_variable_mat(T **mat, unsigned num_rows, unsigned num_cols)
{
    Eigen::Matrix<T, -1, -1> return_disposable;
    return_disposable.resize(num_rows, num_cols);
    uint32_t i = 0, j = 0;

    while(i < num_rows)
    {
        while(j < num_cols)
        {
            return_disposable.row(i).col(j) << mat[i][j];
            j++;
        }
        j = 0;
        i++;
    }
    return return_disposable;
}

template<class T> inline void
Layer<T> :: feed_forward(Layer<T> *&next)
{
    next->set_NeuronArr1D(this->__weight_to_neuron_matMul__());
}

template<class T> void
Layer<T> :: toString()
{
    std::cout << "===================================================\nNeuron Arr" << std::endl;
    for(int i = 0; i < this->num_rows; ++i)
    {
        std::cout << this->get_neuron_val(i) << std::endl;
    }

    std::cout << "\n";
    for(int i = 0; i < this->num_rows; ++i)
    {
        for(int j = 0; j < this->num_cols; ++j)
        {
            std::cout << this->get_variable_val(i, j) << "  ";
        }
        std::cout << std::endl;
    }

    std::cout << "\nWeight Matrix";
    std::cout << "\n===================================================\n\n";
}

template<class T> void
Layer<T> :: toString(int)
{
    std::cout << "===================================================\nNeuron Arr" << std::endl;
    for(int i = 0; i < this->num_rows; ++i)
    {
        std::cout << this->get_neuron_val(i) << std::endl;
    }
    std::cout << "\n===================================================\n\n";
}

#endif /*layer_matrix_hpp*/