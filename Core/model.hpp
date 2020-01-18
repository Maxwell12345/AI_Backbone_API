//
//  model.hpp
//  AI_Backbone
//
//  Created by maxwell on 12/10/2019.
//  Copyright © 2019 organized-organization. All rights reserved.
//

#ifndef model_hpp
#define model_hpp
#include <iostream>
#include "model.cpp"
#include "extern_files.hpp"

template<class T>
Dense<T> :: Dense(bool weight_range, unsigned epochs, bool print, T lr, std::string cost)
{
    this->num_layers = 0;
    this->weight_range = weight_range;
    this->epochs = epochs;
    this->print = print;
    this->network = (Layer<T> **)malloc(sizeof(Layer<T> *));
    this->learning_rate = lr;
    this->cost = cost;
        
    this->mem.learning_rate = this->learning_rate;
    this->mem.cost = this->cost;
}

template<class T> inline void
Dense<T> :: set_input_shape(unsigned int *inp_shape)
{
    this->input_shape = (unsigned int *)calloc(2, sizeof(unsigned int));
    this->input_shape[0] = (inp_shape[0]);
    this->input_shape[1] = (inp_shape[1]);
}

template<class T> inline void
Dense<T> :: initialize_global_variables()
{
    //Use this to initialize the back propogation
    uint32_t i = 1;
    this->network[0] = new Layer<T>(this->lSize_arr[0], this->lSize_arr[1], this->act_func_arr[0], this->weight_range);
    this->network[0]->set_NeuronArr1D(this->input_data[0]);
    this->network[0]->init_Mat2D();
    if(this->bias_boolean_arr[0] == 1)
    {
        this->network[0]->add_bias(this->bias_val_arr[0]);
    }
    while (i < this->num_layers)
    {
        if(i < this->num_layers - 1)
        {
            this->network[i] = new Layer<T>(this->lSize_arr[i], this->lSize_arr[i + 1], this->act_func_arr[i], this->weight_range);
            this->network[i]->init_Mat2D();
            if(this->bias_boolean_arr[i] == 1)
            {
                this->network[i]->add_bias(this->bias_val_arr[i]);
            }
        }
        else
        {
            this->network[i] = new Layer<T>(this->lSize_arr[i], this->act_func_arr[i]);
        }
        
        this->network[i - 1]->feed_forward(this->network[i]);
        i++;
    }
    this->network[this->num_layers - 1]->feed_forward();

    if(this->print)
    {
        this->toCons();
    }
}

template<class T> inline void
Dense<T> :: initialize_network_input(T **inp)
{
    uint32_t i = 0, n = 0;
    Eigen::Matrix<T, -1, 1> set_disposable;
    set_disposable.resize(this->input_shape[1], 1);
    while(n < this->input_shape[0])
    {
        //this->input_data[n].resize(this->input_shape[1], 1);
        while(i < this->input_shape[1])
        {
            set_disposable.row(i).col(0) << inp[n][i];
            i++;
        }
        this->input_data[n].resize(this->input_shape[1], 1);
        this->input_data.push_back(set_disposable);
        set_disposable.setZero();
        n++;
        i = 0;
    }
}

template<class T> inline void
Dense<T> :: initialize_network_input(std::vector<std::vector<T> > inp)
{
    uint32_t i = 0, n = 0;
    Eigen::Matrix<T, -1, 1> set_disposable;
    set_disposable.resize(this->input_shape[1], 1);
    while(n < this->input_shape[0])
    {
        //this->input_data[n].resize(this->input_shape[1], 1);
        while(i < this->input_shape[1])
        {
            set_disposable.row(i).col(0) << inp[n][i];
            i++;
        }
        //this->input_data[n].resize(this->input_shape[1], 1);
        this->input_data.push_back(set_disposable);
        set_disposable.setZero();
        n++;
        i = 0;
    }
}

template<class T> inline void
Dense<T> :: initialize_network_output(T **y_data)
{
    this->mem.set_lArr(this->lSize_arr);
    this->mem.num_sets = this->input_shape[0];
    this->mem.format_y_data(y_data);
}

template<class T> inline void
Dense<T> :: initialize_network_output(std::vector<std::vector<T> > y_data)
{
    this->mem.set_lArr(this->lSize_arr);
    this->mem.num_sets = this->input_shape[0];
    this->mem.format_y_data(y_data);
}

template<class T> inline void
Dense<T> :: add(unsigned lSize, std::string act_func)
{
    this->num_layers += 1;
    this->network = (Layer<T> **)realloc(this->network, sizeof(Layer<T> *) * this->num_layers);
    this->lSize_arr.push_back(lSize);
    this->act_func_arr.push_back(act_func);
}

template<class T> inline void
Dense<T> :: allocate_network_mem()
{
    uint32_t n = 0;
    this->network = (Layer<T> **)malloc(sizeof(Layer<T> *) * this->num_layers);
    while(n < this->num_layers)
    {
        this->network[n] = (Layer<T> *)malloc(sizeof(Layer<T>));
        n++;
    }
}

template<class T> inline void
Dense<T> :: init_BP_network(Layer<T> **net)
{
    uint32_t ii = 0, j = 0;
    T **net_disposable = (T **)calloc(this->num_layers, sizeof(T*));
    while(ii < this->num_layers)
    {
        net_disposable[ii] = (T *)calloc(this->lSize_arr[ii], sizeof(T));
        while(j < this->lSize_arr[ii])
        {
            net_disposable[ii][j] = net[ii]->get_neuron_val(j);
            j++;
        }
        j = 0;
        ii++;
    }
    this->mem.set_network(net_disposable);
    free(net_disposable);
}

template<class T> inline void
Dense<T> :: train()
{
    uint32_t ii = 0;
    Eigen::Matrix<T, -1, -1> *history = (Eigen::Matrix<T, -1, -1> *)calloc(this->lSize_arr.size() - 1, sizeof(Eigen::Matrix<T, -1, -1>));
    Eigen::Matrix<T, -1, 1> *bias_history = (Eigen::Matrix<T, -1, 1> *)calloc(this->lSize_arr.size() - 1, sizeof(Eigen::Matrix<T, -1, 1>));
    std::vector<uint16_t> b_n_arr;

    int data_idx = 1;
    init_mat<T>();
    this->mem.activation_func_arr = this->act_func_arr;
    this->init_BP_network(this->network);
    
    while(ii < this->lSize_arr.size() - 1)
    {
        history[ii].resize(this->lSize_arr[ii], this->lSize_arr[ii+1]);
        history[ii] = this->network[ii]->get_variable_mat();
        
        uint16_t A = this->network[ii]->get_bias_boolean_val();
        if(this->network[ii]->get_bias_boolean_val() == 1)
        {
            bias_history[ii].resize(this->lSize_arr[ii], 1);
            bias_history[ii] = this->network[ii]->get_bias_mat();
        }
        b_n_arr.push_back(this->network[ii]->get_bias_boolean_val());
        this->mem.b_n_arr.push_back(b_n_arr[ii]);

        ii++;
    }ii=0;
    this->mem.record_mat_data(history);
    this->mem.set_bias(bias_history);
    this->mem.update_network_variables();
    this->mem.update_network_bias();
    

    for(int n = 1; n < this->epochs; ++n)
    {
        this->network[0]->set_NeuronArr1D(this->input_data[data_idx]);
        if(data_idx == this->input_shape[0] - 1)
        {
            data_idx = 0;
        }
        else
        {
            data_idx++;
        }
        this->network[0]->set_Mat2D(Layer<T>::format_variable_mat(__W_Mat_Mem__<T>[/*n - 1*/ 0][0], this->lSize_arr[0], this->lSize_arr[1]));
        this->network[0]->set_bias_arr(__W_Bias_Mem__<T>[/*n - 1*/ 0][0]); // FIXX
        this->network[0]->feed_forward(this->network[1]);

        for(int i = 1; i < this->num_layers - 1; ++i)
        {
            this->network[i]->set_Mat2D(Layer<T>::format_variable_mat(__W_Mat_Mem__<T>[/*n - 1*/ 0][i], this->lSize_arr[i], this->lSize_arr[i + 1]));
            if(this->bias_boolean_arr[i] == 1){this->network[i]->set_bias_arr(__W_Bias_Mem__<T>[/*n - 1*/ 0][i]);}
            this->network[i]->feed_forward(this->network[i + 1]);
        }this->network[this->num_layers - 1]->feed_forward();

        if(this->print)
        {
            this->toCons();
        }
        init_mat<T>();
        while(ii < this->num_layers - 1)
        {
            history[ii].resize(this->lSize_arr[ii], this->lSize_arr[ii+1]);
            history[ii] = this->network[ii]->get_variable_mat();

            if(this->network[ii]->get_bias_boolean_val())
            {
                bias_history[ii].resize(this->lSize_arr[ii], 1);
                bias_history[ii] = this->network[ii]->get_bias_mat();
            }
            ii++;
        }ii=0;
        this->mem.record_mat_data(history);
        this->init_BP_network(this->network);
        this->mem.set_bias(bias_history);
        this->mem.update_network_variables();
        this->mem.update_network_bias();
    }
    free(history);
}

template<class T> inline void
Dense<T> :: bias(T b_val)
{
    uint32_t i = 0, b_vec_size = this->bias_val_arr.size();
    if(this->num_layers != (b_vec_size + 1))
    {
        do
        {
            this->bias_val_arr.push_back(0);
            this->bias_boolean_arr.push_back(0);
            i++;
        }while(i <= this->num_layers - (b_vec_size + 1));
    }
    this->bias_val_arr.push_back(b_val);
    this->bias_boolean_arr.push_back(1);
}

template<class T> void
Dense<T> :: toCons()
{
    iteration++;
    std::cout << "\n\n" << iteration << std::endl;
    for(int i = 0; i < this->num_layers - 1; ++i)
    {
        this->network[i]->toString();
    }
    this->network[this->num_layers - 1]->toString(0);

}

#endif /*model_hpp*/