#ifndef REDUCTION_TOKEN_HPP
#define REDUCTION_TOKEN_HPP

#include "kernel_generator.h"
#include <cmath>

class total_BMT_result_reduce_to_one_register_token : public reduction_basic_token
{
public:
    total_BMT_result_reduce_to_one_register_token(shared_ptr<meta_data_set> meta_data_set_ptr, bool need_warp_reduction, unsigned int sparse_coarsen_factor, unsigned int coarsen_factor, shared_ptr<code_generator> code_generator_ptr);
    shared_ptr<basic_IO_of_reduction> get_output_IO();
    string run();
private:
    bool need_warp_reduction;
    unsigned int coarsen_factor;
    unsigned int sparse_coarsen_factor;
    shared_ptr<basic_IO_of_reduction> input_IO = NULL;
    shared_ptr<basic_IO_of_reduction> output_IO = NULL;
};

// class total_BMT_result_reduce_to_one_register_with_shared_sparse_token : public reduction_basic_token
// {
// public:
//     total_BMT_result_reduce_to_one_register_with_shared_sparse_token(shared_ptr<meta_data_set> meta_data_set_ptr, unsigned int sparse_coarsen_factor, unsigned int coarsen_factor, shared_ptr<code_generator> code_generator_ptr);
//     shared_ptr<basic_IO_of_reduction> get_output_IO();
//     string run();
// private:
//     unsigned int coarsen_factor;
//     unsigned int sparse_coarsen_factor;
//     shared_ptr<basic_IO_of_reduction> input_IO = NULL;
//     shared_ptr<basic_IO_of_reduction> output_IO = NULL;
// };


class thread_bit_map_reduce_to_two_register_token : public reduction_basic_token
{
public:
    thread_bit_map_reduce_to_two_register_token(shared_ptr<meta_data_set> meta_data_set_ptr, bool need_warp_reduction, unsigned int sparse_coarsen_factor, unsigned int coarsen_factor, shared_ptr<code_generator> code_generator_ptr);
    shared_ptr<basic_IO_of_reduction> get_output_IO();
    string run();
private:
    bool need_warp_reduction;
    unsigned int sparse_coarsen_factor;
    unsigned int coarsen_factor;
    shared_ptr<basic_IO_of_reduction> input_IO = NULL;
    shared_ptr<basic_IO_of_reduction> output_IO = NULL;
};


class total_warp_result_reduce_to_one_register_token : public reduction_basic_token
{
public:
    total_warp_result_reduce_to_one_register_token(shared_ptr<meta_data_set> meta_data_set_ptr, unsigned int coarsen_factor, shared_ptr<code_generator> code_generator_ptr);
    shared_ptr<basic_IO_of_reduction> get_output_IO();
    string run();
private:
    unsigned int coarsen_factor;
    shared_ptr<basic_IO_of_reduction> input_IO = NULL;
    shared_ptr<basic_IO_of_reduction> output_IO = NULL;
};


class warp_bit_map_reduce_token : public reduction_basic_token
{
public:
    warp_bit_map_reduce_token(shared_ptr<meta_data_set> meta_data_set_ptr, unsigned int coarsen_factor, shared_ptr<code_generator> code_generator_ptr);
    shared_ptr<basic_IO_of_reduction> get_output_IO();
    string run();
private:
    unsigned int coarsen_factor;
    shared_ptr<basic_IO_of_reduction> input_IO = NULL;
    shared_ptr<basic_IO_of_reduction> output_IO = NULL;
};

class warp_segment_reduce_token : public reduction_basic_token
{
public:
    warp_segment_reduce_token(shared_ptr<meta_data_set> meta_data_set_ptr, unsigned int coarsen_factor, shared_ptr<code_generator> code_generator_ptr);
    shared_ptr<basic_IO_of_reduction> get_output_IO();
    string run();
private:
    unsigned int coarsen_factor;
    shared_ptr<basic_IO_of_reduction> input_IO = NULL;
    shared_ptr<basic_IO_of_reduction> output_IO = NULL;
};


class total_block_result_reduce_to_one_register_token : public reduction_basic_token
{
public:
    total_block_result_reduce_to_one_register_token(shared_ptr<meta_data_set> meta_data_set_ptr, unsigned int coarsen_factor, shared_ptr<code_generator> code_generator_ptr);
    shared_ptr<basic_IO_of_reduction> get_output_IO();
    string run();
private:
    unsigned int coarsen_factor;
    shared_ptr<basic_IO_of_reduction> input_IO = NULL;
    shared_ptr<basic_IO_of_reduction> output_IO = NULL;
};

class tblock_bit_map_reduce_token : public reduction_basic_token
{
public:
    tblock_bit_map_reduce_token(shared_ptr<meta_data_set> meta_data_set_ptr, unsigned int coarsen_factor, shared_ptr<code_generator> code_generator_ptr);
    shared_ptr<basic_IO_of_reduction> get_output_IO();
    string run();
private:
    unsigned int coarsen_factor;
    shared_ptr<basic_IO_of_reduction> input_IO = NULL;
    shared_ptr<basic_IO_of_reduction> output_IO = NULL;
};






#endif