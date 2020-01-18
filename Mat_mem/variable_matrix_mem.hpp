//
//  variable_amtrix_mem.hpp
//  AI_Backbone
//
//  Created by maxwell on 12/10/2019.
//  Copyright © 2019 organized-organization. All rights reserved.
//
#include <vector>
#include "BP_extern_files.hpp"
static unsigned id = 1;

template<class T>  // This variable is then updated each epoch iteration!!
T ****__W_Mat_Mem__ ;

template<class T>
T ***__W_Bias_Mem__ = (T ***)malloc(sizeof(T ***));

template<class T> static void 
set_mat_mem(Eigen::Matrix<T, -1, -1> *history, std::vector<unsigned> lSize_arr)
{
    uint32_t i = id, j = 0, n = 0, k = 0;
    // i = the whole network index
    // j = the layer index in the network
    // n = the row index
    // k = the col index
    __W_Mat_Mem__<T> = (T ****)realloc(__W_Mat_Mem__<T>, sizeof(T ***) * id);
    while(i < id + 1)
    {
        __W_Mat_Mem__<T>[id - 1] = (T ***)calloc(lSize_arr.size(), (sizeof(T **)));
        while(j < lSize_arr.size() - 1)
        {
            __W_Mat_Mem__<T>[id - 1][j] = (T **)calloc(lSize_arr[j], (sizeof(T *)));
            while(n < lSize_arr[j])
            {
                __W_Mat_Mem__<T>[id - 1][j][n] = (T *)calloc(lSize_arr[j + 1], (sizeof(T)));
                while(k < lSize_arr[j + 1])
                {
                    __W_Mat_Mem__<T>[id - 1][j][n][k] = history[j](n, k);
                    k++;
                }
                k = 0;
                n++;
            }
            n = 0;
            j++;
        }
        j = 0;
        i++;
    }

    //id++;
}

template<class T> static void 
set_bias_mem(Eigen::Matrix<T, -1, 1> *bias_history, std::vector<unsigned> lSize_arr, std::vector<uint16_t> b_n_arr)
{
    uint32_t i = 0, j = 0, n = 0;
    __W_Bias_Mem__<T> = (T ***)realloc(__W_Bias_Mem__<T>, sizeof(T **) * id);
    __W_Bias_Mem__<T>[id - 1] = (T **)calloc(lSize_arr.size(), sizeof(T *));
    while(i < lSize_arr.size() - 1)
    {
        __W_Bias_Mem__<T>[id - 1][i] = (T *)calloc(lSize_arr[i+1], sizeof(T));

        if(b_n_arr[i] == 1)
        {
            while(j < lSize_arr[i + 1])
            {
                __W_Bias_Mem__<T>[id - 1][i][j] = bias_history[i](j, 0);
                j++;
            }
        }
        j = 0;
        i++;
    }
}

