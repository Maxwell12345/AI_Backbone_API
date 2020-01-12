//
//  back_proagation.hpp
//  AI_Backbone
//
//  Created by maxwell on 12/10/2019.
//  Copyright © 2019 organized-organization. All rights reserved.
//

#ifndef back_propagation_hpp
#define back_propagation_hpp
#include <cmath>
#include "back_propagation.cpp"
// #include "BP_extern_files.hpp"

template<class T> class
PtrAutoDispose 
{ 
    T *ptr;  // Actual pointer 
public:
   explicit PtrAutoDispose(T *p = NULL) { ptr = p; } 
   ~PtrAutoDispose() { delete(ptr); } 
  
   // Overloading dereferncing operator 
   T & operator * () {  return *ptr; } 
   T * operator -> () { return ptr; } 
}; 

template<class T> inline void 
init_mat(void) // Use in the train method for each model class
{
    if(id == 1)
    {
        __W_Mat_Mem__<T> = (T ****)malloc(sizeof(T ****));
    }
    // __W_Mat_Mem__<T>[id - 1] = (T***)malloc(sizeof(T **));
    // __W_Mat_Mem__<T>[id - 1][0] = (T**)malloc(sizeof(T *));
    // __W_Mat_Mem__<T>[id - 1][0][0] = (T *)malloc(sizeof(T));
};

template<typename T> inline void
_update_variable_mat_<T> :: record_mat_data(Eigen::Matrix<T, -1, -1> *history)
{
    set_mat_mem(history, this->lSize_arr);
}

template<class T> inline void
_update_variable_mat_<T> :: set_lArr(std::vector<unsigned> lArr)
{
    this->lSize_arr = lArr;
}

template<class T> inline void
_update_variable_mat_<T> :: format_y_data(T **y_data)
{
    uint32_t i = 0, j = 0;
    
    Eigen::Matrix<T, -1, 1> set_disposable;
    set_disposable.resize(this->lSize_arr[this->lSize_arr.size() - 1], 1);

    while(i < this->num_sets)
    {
        while(j < this->lSize_arr[this->lSize_arr.size() - 1])
        {
            set_disposable.row(j).col(0) << y_data[i][j];
            j++;
        }
        this->y_data.push_back(set_disposable);
        this->y_data[i].resize(this->lSize_arr[this->lSize_arr.size() - 1], 1);
        set_disposable.setZero();
        j = 0;
        i++;
    }
}

template<class T> inline void
_update_variable_mat_<T> :: format_y_data(std::vector<std::vector<T> > y_data)
{
    uint32_t i = 0, j = 0;
    
    Eigen::Matrix<T, -1, 1> set_disposable;
    set_disposable.resize(this->lSize_arr[lSize_arr.size() - 1], 1);

    while(i < this->num_sets)
    {
        while(j < this->lSize_arr[this->lSize_arr.size() - 1])
        {
            set_disposable.row(j).col(0) << y_data[i][j];
            j++;
        }
        this->y_data.push_back(set_disposable);
        this->y_data[i].resize(this->lSize_arr[this->lSize_arr.size() - 1], 1);
        set_disposable.setZero();
        j = 0;
        i++;
    }
}

template<class T> inline T
_update_variable_mat_<T> :: get_error_val(T x, uint32_t index)
{
    if(this->cost == "MeanSqrErr") { return (T)pow(x - this->y_data[this->epoch_idx](index, 0), 2); }
    else if(this->cost == "MeanAbsErr") { return (T)abs(x - this->y_data[this->epoch_idx](index, 0)); }
    else if(this->cost == "cat_crossentropy") { return (T)-1 * (T)log(x) * this->y_data[this->epoch_idx](index, 0); }
    // categorical crossentropy (For when the prob outputs ARE NOT binary)
    else
    {
        std::cout << "Case Error! No Cost Function With Name: " << this->cost << std::endl;
        return -1;
    }
}

template<class T> inline T
_update_variable_mat_<T> :: get_error_val(Eigen::Matrix<T, -1, 1> outps, bool get_sumation, uint32_t index)
{
    uint32_t i = index;
    uint32_t _error = 0;
    T n;
    if(get_sumation)
        n = outps.rows();
    else
        n = index + 1;
    
    if(this->cost == "MeanSqrErr")
    {
        while(i < n)
        {
            _error += (T)pow((outps(i, 0) - this->y_data[this->epoch_idx](i, 0)), 2);
        }
        return _error;
    }
    else if(this->cost == "MeanAbsErr")
    {
        while(i < n)
        {
            _error += (T)abs(outps(i, 0) - this->y_data[this->epoch_idx](i, 0));
        }
        return _error;
    }
    else if(this->cost == "cat_crossentropy") 
    // categorical crossentropy (For when the prob outputs ARE NOT binary)
    {
        while(i < n)
        {
            _error += (T)log(outps(i, 0)) * this->y_data[this->epoch_idx](i, 0);
        }
        return -1 * _error;
    }
    // else if(this->cost == "bin_crossentropy") 
    // // categorical crossentropy (For when the prob outputs ARE binary)
    // {
    //     while(i < n)
    //     {
    //         _error += (T)log(outps(i, 0)) * this->y_data[this->epoch_idx](i, 0);
    //     }
    //     return _error;
    // }
    else
    {
        std::cout << "Case Error! No Cost Function With Name: " << this->cost << std::endl;
        return -1;
    }
    
}

template<class T> inline T
_update_variable_mat_<T> :: dirivative(T *x_arr, T neuronOutp, unsigned num_layers_to_backproagate, unsigned oNeuron_idx)
{
    uint32_t nLayers = 0;
    T weight_mutator = this->get_error_val(neuronOutp, oNeuron_idx);

    while(nLayers < num_layers_to_backproagate)
    {
        weight_mutator *= __activation_func_derivatives__(x_arr[nLayers], 
            this->activation_func_arr[this->lSize_arr.size() - nLayers - 1]);
        nLayers ++;
    }
    
    return weight_mutator;
}

template<class T> inline T
_update_variable_mat_<T> :: dirivative(T *x_arr, uint32_t update_idx)
{
    uint32_t nLayers = 0;
    T weight_mutator = 1;

    while(nLayers < 2)
    {
        weight_mutator *= __activation_func_derivatives__(x_arr[nLayers], this->activation_func_arr[update_idx + 1 - nLayers]);
        nLayers++;
    }

    return weight_mutator;
}

template<class T> inline void
_update_variable_mat_<T> :: update_network_variables(Eigen::Matrix<T, -1, 1> neuronOutps)
{
    //ptr array that hols the data regarding each neuron value
    T *x_arr_disposable = (T *)calloc(2, sizeof(T));
    
    /* If you are going to optimize algorithm declare and initialize with the following code */
    //uint32_t _count = 1, nLayers = 0;
    //This basically holds the un learingrate returned chain list product
    //e.g. (cost dir) * (layer 1 dir) * (layer 2 dir) * lr
    // PtrAutoDispose<T> *dirivative_optimizer(new T());

    // while(nLayers < this->lSize_arr.size())
    // {
    //     _count *= this->lSize_arr[this->lSize_arr.size() - nLayers - 1];
    //     dirivative_optimizer = (PtrAutoDispose<T> *)calloc(_count, sizeof(PtrAutoDispose<T>));
    //     nLayers++;
    // }nLayers = 0; _count = 1;
    neuronOutps.resize(this->lSize_arr[this->lSize_arr.size() - 1], 1);
        
    for(int n = this->lSize_arr.size() - 1; n > 0; --n)
    {
        if(n == this->lSize_arr.size() - 1)
        {
            for(int i = 0; i < this->lSize_arr[n - 1]; ++i)
            {
                for(int j = 0; j < this->lSize_arr[n]; ++j)
                {
                    x_arr_disposable[0] = __W_Mat_Mem__<T>[id - 1][n - 1][i][j];
                    
                    __W_Mat_Mem__<T>[id - 1][n - 1][i][j] = 
                    (__W_Mat_Mem__<T>[id - 1][n - 1][i][j] - this->dirivative(x_arr_disposable, neuronOutps(i, 0), 1, i)) * this->learning_rate;
                }
            }
        }
        else
        {
            for(int o = 0; o < this->lSize_arr[n - 1]; ++o)
            {
                for(int i = 0; i < this->lSize_arr[n]; ++i)
                {
                    for(int j = 0; j < this->lSize_arr[n + 1]; ++j)
                    {
                        for(int c = 0; c < 2; ++c)
                        {
                            if(c == 0)
                            {
                                x_arr_disposable[c] = __W_Mat_Mem__<T>[id - 1][n - c][i][j];
                            }
                            else
                            {
                                x_arr_disposable[c] = __W_Mat_Mem__<T>[id - 1][n - c][o][i];
                            }
                        }

                        __W_Mat_Mem__<T>[id - 1][n - 1][o][i] = 
                        (__W_Mat_Mem__<T>[id - 1][n - 1][o][i] - this->dirivative(x_arr_disposable, n - 1)) * this->learning_rate;
                    }
                }
            }
        } 
    }

    // for(int i = 0; i < this->lSize_arr[this->lSize_arr.size()] - 2; ++i)
    // {
    //     for(int j = 0; j < this->lSize_arr[i]; ++j)
    //     {
    //         T b = __W_Mat_Mem__<T>[id - 1][0][1][0];
    //         for(int n = 0; n < this->lSize_arr[i + 1]; ++i)
    //         {
    //             __W_Mat_Mem__<T>[id - 1][i][j][n] =__W_Mat_Mem__<T>[id - 1][i][j][n] *  this->learning_rate;
    //         }
    //     }
    // }
    
    free(x_arr_disposable);
}

#endif /*back_propagation_hpp*/