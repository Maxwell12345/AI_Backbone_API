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
        set_disposable.setZero();
        j = 0;
        i++;
    }
}

template<class T> inline T
_update_variable_mat_<T> :: get_error_val(T x, uint32_t index)
{
    this->y_data[index].resize(this->lSize_arr[this->lSize_arr.size() - 1], 1);
    if(this->cost == "MeanSqrErr") { return (x - this->y_data[this->epoch_idx](index, 0)); } // Get dirvative!!
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

// template<class T> inline T
// _update_variable_mat_<T> :: dirivative(T *x_arr, unsigned num_layers_to_backproagate, unsigned oNeuron_idx)
// {
//     uint32_t nLayers = 0;
//     T weight_mutator = this->get_error_val(neuronOutp, oNeuron_idx); // ERROR IN ERR_CATCH_FILE

//     while(nLayers < num_layers_to_backproagate)
//     {
//         weight_mutator *= __activation_func_derivatives__(x_arr[nLayers], 
//             this->activation_func_arr[this->lSize_arr.size() - nLayers - 1]);
//         nLayers ++;
//     }
    
//     return weight_mutator;
// }

template<class T> inline T
_update_variable_mat_<T> :: dirivative(T x_val, uint32_t update_idx)
{
    T weight_mutator = __activation_func_derivatives__(x_val, this->activation_func_arr[update_idx]);

    return weight_mutator;
}

template<class T> inline T
_update_variable_mat_<T> :: dirivative(T x_val, T nVal, uint32_t oNeuron_idx)
{
    return (this->get_error_val(x_val, oNeuron_idx) * __activation_func_derivatives__(x_val, this->activation_func_arr[this->lSize_arr.size() - 1]));
}



template<class T>inline void
_update_variable_mat_<T> :: set_network(T **real_network)
{
    uint32_t i = 0, j = 0;
    std::vector<T> set_disposable;
    while(i < this->lSize_arr.size())
    {
        while(j < this->lSize_arr[i])
        {
            set_disposable.push_back(real_network[i][j]);
            j++;
        }
        this->pseudo_network.push_back(set_disposable);
        set_disposable.clear();
        j = 0;
        i++;
    }
}

template<class T> inline void
_update_variable_mat_<T> :: update_network_variables()
{
    //ptr array that hols the data regarding each neuron value
    T *x_arr_disposable = (T *)malloc(2 * sizeof(T));
    T *error_mat = (T *)malloc(sizeof(T) * this->lSize_arr[this->lSize_arr.size() - 1]);
    T val_disposable;

    T ***mat_disposable = (T ***)calloc(this->lSize_arr.size() - 1, sizeof(T**));
    for(int n = 0; n < this->lSize_arr.size() - 1; ++n)
    {
        mat_disposable[n] = (T**)calloc(this->lSize_arr[n], sizeof(T*));
        for(int i = 0; i < this->lSize_arr[n]; ++i){mat_disposable[n][i] = (T*)calloc(this->lSize_arr[n+1], sizeof(T));}
    }

        
    // for(int n = this->lSize_arr.size() - 2; n >= 0; --n)
    // {
    //     if(n == this->lSize_arr.size() - 2)
    //     {
    //         for(int i = 0; i < this->lSize_arr[n]; ++i)
    //         {
    //             T t = this->pseudo_network[n][i];
    //             x_arr_disposable[0] = pseudo_network[n][i];
    //             for(int j = 0; j < this->lSize_arr[n+1]; ++j)
    //             {
    //                 x_arr_disposable[1] = pseudo_network[n+1][j];

    //                 //__W_Mat_Mem__<T>[id - 1][n][i][j] = (__W_Mat_Mem__<T>[id - 1][n][i][j] - 
    //                 mat_disposable[n][i][j] = (this->dirivative(x_arr_disposable[0], x_arr_disposable[1], j));
    //             }
    //         }
    //     }
    //     else
    //     {
    //         // free(x_arr_disposable);
    //         for(int i = 0; i < this->lSize_arr[n]; ++i)
    //         {
    //             val_disposable = pseudo_network[n][i];
    //             for(int j = 0; j < this->lSize_arr[n+1]; ++j)
    //             {
    //                 //__W_Mat_Mem__<T>[id - 1][n][i][j] = (__W_Mat_Mem__<T>[id - 1][n][i][j] - 
    //                 mat_disposable[n][i][j] = (this->dirivative(val_disposable, n) * __W_Mat_Mem__<T>[id - 1][n+1][j][i]);
    //             }
    //         }
    //     }
    // }

    // Get single cost value
    // get dirivative of actication function of output neuron
    // this creates an error matrix
    for(int i = 0; i < this->lSize_arr[this->lSize_arr.size() - 1]; ++i)
    {
        error_mat[i] = get_error_val(this->pseudo_network[this->lSize_arr.size() - 1][i], i);
        error_mat[i] *= __activation_func_derivatives__(this->pseudo_network[this->lSize_arr.size() - 1][i], this->activation_func_arr[this->lSize_arr.size() - 1]);
    }
    // these values are matrix multiplied with the activated values of the previous layer
    for(int i = 0; i < this->lSize_arr[this->lSize_arr.size() - 2]; ++i)
    {
        for(int j = 0; j < this->lSize_arr[this->lSize_arr.size() - 1]; ++j)
        {
            __W_Mat_Mem__<T>[id - 1][this->lSize_arr.size() - 2][i][j] -= __W_Mat_Mem__<T>[id - 1][this->lSize_arr.size() - 2][i][j] * 
            error_mat[j] * this->pseudo_network[this->lSize_arr.size() - 2][i] * this->learning_rate;
        }
    }

    for(int i = 0; i < this->lSize_arr[0]; ++i)
    {
        for(int j = 0; j < this->lSize_arr[this->lSize_arr.size() - 2]; ++j)
        {
            for(int n = 0; n < this->lSize_arr[this->lSize_arr.size() - 1]; ++n)
            {
                __W_Mat_Mem__<T>[id - 1][0][i][j] -= __W_Mat_Mem__<T>[id - 1][0][i][j] * 
                error_mat[n] * __activation_func_derivatives__(this->pseudo_network[1][j], activation_func_arr[1]) * 
                this->pseudo_network[0][i] * learning_rate;
            }
        }
    }
    

    // for(int n = 0; n < this->lSize_arr.size() - 1; ++n)
    // {
    //     for(int i = 0; i < this->lSize_arr[n]; ++i)
    //     {
    //         for(int j = 0; j < this->lSize_arr[n+1]; ++j)
    //         {
    //             __W_Mat_Mem__<T>[id - 1][n][i][j] = __W_Mat_Mem__<T>[id - 1][n][i][j] - 
    //                 mat_disposable[n][i][j] * this->learning_rate;
    //         }
    //     }
    // }
}

#endif /*back_propagation_hpp*/