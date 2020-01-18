#include <iostream>
#include <vector>
#include "BP_extern_files.hpp"
#include "variable_matrix_mem.hpp"

template<class T>
struct _update_variable_mat_
{
    // These need to be initiated first before any instance variable or method is used
    std::vector<unsigned> lSize_arr;

    std::vector<Eigen::Matrix<T, -1, 1> >y_data;
    std::string cost;
    std::vector<std::string> activation_func_arr;
    std::vector<uint16_t> b_n_arr;
    uint32_t num_sets;
    T learning_rate;
    uint32_t epoch_idx;

    inline void record_mat_data(Eigen::Matrix<T, -1, -1> *history);
    inline T get_error_val(Eigen::Matrix<T, -1, 1> outps, bool get_sumation, uint32_t index); // index is 0 if sumation is true
    inline T get_error_val(T x, uint32_t index);
    inline T dirivative(T *, unsigned num_layers_to_backproagate, unsigned oNeuron_idx); 
    inline T dirivative(T, uint32_t update_idx); //update_idx is the index of the deepest mat being updated
    inline T dirivative(T, T, uint32_t oNeuron_idx);
    // The pointer array goes in order from the output neuron to the weight that needs to be update
    inline void format_y_data(T **);
    inline void format_y_data(std::vector<std::vector<T> >);
    inline void update_network_variables(void);
    inline void update_network_bias(void);
    inline void set_lArr(std::vector<unsigned>);
    inline void set_network(T**);
    inline void set_bias(Eigen::Matrix<T, -1, 1> *bias_history);

    std::vector<std::vector<T> >pseudo_network;
};