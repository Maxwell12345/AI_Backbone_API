//
//  activation_data.hpp
//  AI_Backbone
//
//  Created by maxwell on 12/10/2019.
//  Copyright © 2019 organized-organization. All rights reserved.
//

#ifndef activation_data_hpp
#define activation_data_hpp

#include <iostream>
#include <string>
#include <cmath>

template<class T> inline T 
__ret_activated_val__(T x, std::string act_func)
{
    if(act_func == "sigmoid")
    {
        return (T)(1 / (1 + (T)exp(x* -1)));
    }
    else if(act_func == "relu")
    {
        if(x < 0) return 0;
        else return x;
    }
    else if(act_func == "tanh")
    {
        return (T)tanh(x);
    }
    else if(act_func == "atan")
    {
        return (T)atan(x);
    }
    else if(act_func == "log")
    {
        return (T)log10(x);
    }
    else if(act_func == "leaky_relu")
    {
        if(x < 0) return x * (T)0.1;
        else return x;
    }
    else if(act_func == "linear")
    {
        return x;
    }
    else if(act_func == "asinh")
    {
        return (T)asinh(x);
    }
    else
    {
        std::cout << "No Activation Function With Name " + act_func << std::endl;
        return -1;
    }
}

template<class T> inline T
__ret_activated_val__(T *x, std::string act_func, uint32_t num_oNeurons, uint32_t idx)
{
    uint32_t i = 0;
    T rSum = 0;
    if(act_func == "softmax")
    {
        while(i < num_oNeurons)
        {
            rSum += (T)exp(x[i]);
            i++;
        }
        return (T)exp(x[idx]) / rSum;
    }
    else if(act_func == "hardmax")
    {
        while(i < num_oNeurons - 1)
        {
            if(x[i] > x[i+1])
            {
                rSum = x[i];
            }
            else
            {
                rSum = x[i+1];
            }
            i++;
        }
        if(rSum == x[idx])
        {
            return 1;
        }
        else
        {
            return 0;
        }
    }
}

//This return value is based on the notion that the inputed (x) is already activated
//e.g. 1/1+e^-x = y,  y(1-y) = y'
template<class T> inline T 
__activation_func_derivatives__(T x, std::string act_func)
{
    if(act_func == "sigmoid")
    {
        return x * ((T)1 - x);
    }
    else if(act_func == "relu")
    {
        return 1;
    }
    else if(act_func == "tanh")
    {
        return 1 - (T)pow(x, 2);
    }
    else if(act_func == "atan")
    {
        return (T)(1 / (1 + (T)pow((T)tan(x), 2)));
    }
    else if(act_func == "log")
    {
        //IDK HOW TO GET THE DIRIVATIVE WITH RESPECT TO LOG
        return x;
    }
    else if(act_func == "leaky_relu")
    {
        if(x < 0) return (T)(0.1);
        else return 1;
    }
    else if(act_func == "linear")
    {
        return 1;
    }
    else if(act_func == "asinh")
    {
        return (T)(1 / (T)(cosh(x)));
    }
    else
    {
        std::cout << "No Activation Function Dirivated With Name " + act_func << std::endl;
        return -1;
    }
}

#endif /*activation_data_hpp*/