#include "../reduction_token.hpp"
#include "../IO_of_reduction.hpp"
#include "../code_generator.hpp"

warp_bit_map_reduce_token::warp_bit_map_reduce_token(shared_ptr<meta_data_set> meta_data_set_ptr, unsigned int coarsen_factor, shared_ptr<code_generator> code_generator_ptr)
    : reduction_basic_token(WARP_META, meta_data_set_ptr)
{
    assert(meta_data_set_ptr != NULL);
    assert(code_generator_ptr != NULL);
    assert(code_generator_ptr->check() == true);

    this->code_generator_ptr = code_generator_ptr;
    this->coarsen_factor = coarsen_factor;

    int count = coarsen_factor;

    shared_ptr<one_register_result_IO_of_reduction> one_register_IO_in(new one_register_result_IO_of_reduction(THREAD_META, count));
    this->input_IO = one_register_IO_in;

    
    shared_ptr<one_register_result_IO_of_reduction> one_register_IO_out(new one_register_result_IO_of_reduction(WARP_META, count));
    this->output_IO = one_register_IO_out;
}

// 执行对应的代码生成过程
string warp_bit_map_reduce_token::run()
{
    // 将代码生成器复原成智能指针
    shared_ptr<code_generator> code_generator_ptr = this->code_generator_ptr.lock();

    if (code_generator_ptr == NULL)
    {
        cout << "warp_bit_map_reduce_token::run(): the code generator has already destroyed" << endl;
        assert(false);
    }

    assert(code_generator_ptr->check() == true);

    stringstream return_code_str;

    string input_result_name = dynamic_pointer_cast<one_register_result_IO_of_reduction>(this->input_IO)->var_name_token_of_IO_register()->run();
    string result_register_name = dynamic_pointer_cast<one_register_result_IO_of_reduction>(this->output_IO)->var_name_token_of_IO_register()->run();


    int sub_matrix_id = code_generator_ptr->get_sub_matrix_id();

    assert(sub_matrix_id >= 0);

    assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_vals", sub_matrix_id) == true);
    shared_ptr<universal_array> var_array_ptr = this->meta_data_set_ptr
                                                    ->get_element(GLOBAL_META, "nz_vals", sub_matrix_id)
                                                    ->get_metadata_arr();

    data_type data_type_of_result = var_array_ptr->get_data_type();
    if (data_type_of_result == FLOAT && get_config()["HALF"].as_bool() == true)
    {
        data_type_of_result = HALF;
    }
    shared_ptr<math_expr_token> dense_matrix_ptr_init = NULL;
    shared_ptr<math_expr_token> dense_matrix_step = NULL;

    string dense_ = "((blockIdx.y * blockDim.y) + threadIdx.y) * " + to_string(this->coarsen_factor);
    dense_matrix_ptr_init = make_shared<math_expr_token>(dense_);
    dense_matrix_step = make_shared<math_expr_token>("blockDim.y * gridDim.y * " + to_string(this->coarsen_factor));
    
    shared_ptr<math_expr_token> lane_id_init(new math_expr_token("threadIdx.x % " + to_string((int)get_config()["VECTOR_WIDTH"].as_float())));
    shared_ptr<var_name_token> result_register = code_generator_ptr->generate_global_var(data_type_of_result, result_register_name, NULL, this->coarsen_factor);
    shared_ptr<var_name_token> dense_matrix_ptr = code_generator_ptr->generate_global_var(UNSIGNED_INT, "dense_matrix_ptr", dense_matrix_ptr_init);
    shared_ptr<var_name_token> lane_id = code_generator_ptr->generate_global_var(UNSIGNED_CHAR, "lane_id", lane_id_init);

    shared_ptr<math_expr_token> cur_BMW_id_token(new math_expr_token(code_generator_ptr->code_of_data_block_id_distributed_to_spec_paral(WARP_META)->run()));
    shared_ptr<math_expr_token> next_BMW_id_token(new math_expr_token(code_generator_ptr->code_of_data_block_id_distributed_to_spec_paral(WARP_META)->run() + "+1"));
    shared_ptr<var_name_token> warp_bit_map = code_generator_ptr->generate_fused_memory_access(WARP_META, "bit_map_of_thread", cur_BMW_id_token);

    string row_name_of_thread;

    if(this->meta_data_set_ptr->is_exist(THREAD_META, "first_row_indices", sub_matrix_id) == true)
    {
        row_name_of_thread = "first_row_indices";
    }
    else
    {
        row_name_of_thread = "first_row_indices_without_ending";
    }

    data_type nz_data_type = this->meta_data_set_ptr->get_element(THREAD_META, row_name_of_thread, sub_matrix_id)
                                 ->get_metadata_arr()
                                 ->get_data_type();
    
    string dense_ptr = dense_matrix_ptr->run();



    return_code_str << code_of_data_type(data_type_of_result) << " temp_result[" << to_string(this->coarsen_factor) << "];" << endl;
    return_code_str << code_of_data_type(nz_data_type) << " temp_row = 0;" << endl;

    if (this->coarsen_factor > 1)
    {
        return_code_str << "for (unsigned int p = 0; p <" << to_string(this->coarsen_factor) << "; p++)" << endl;
        return_code_str << "{" << endl;
        return_code_str << result_register_name << "[p] = " << input_result_name << "[p];" << endl;
        return_code_str << "}" << endl;
    }
    else
    {
        return_code_str << result_register_name << "[0] = " << input_result_name << "[0];" << endl;
    }

    return_code_str << "if(" << dense_ptr << " + " << to_string(this->coarsen_factor) << " <= K)";
    return_code_str << "{" << endl;

    if (this->coarsen_factor > 1)
    {
        if (get_config()["VECTOR_WIDTH"].as_float() >= 32)
        {
            return_code_str << "temp_row = __shfl_down_sync(0xffffffff, first_row_thread, 16);" << endl;
            return_code_str << "for(unsigned int r = 0; r <" << to_string(this->coarsen_factor) << "; r++)" << endl;
            return_code_str << "{" << endl;
            return_code_str << "temp_result[r] =  __shfl_down_sync(0xffffffff, " << result_register_name << "[r], 16);" << endl;
            return_code_str << "}" << endl;
            return_code_str << "if (first_row_thread == temp_row && lane_id < 16){" << endl;
            return_code_str << "for(unsigned int r = 0; r <" << to_string(this->coarsen_factor) << "; r++)" << endl;
            return_code_str << "{" << endl;
            return_code_str << result_register_name << "[r] += temp_result[r];" << endl;
            return_code_str << "}" << endl;
            return_code_str << "}" << endl;
        }

        if (get_config()["VECTOR_WIDTH"].as_float() >= 16)
        {
            return_code_str << "temp_row = __shfl_down_sync(0xffffffff, first_row_thread, 8);" << endl;
            return_code_str << "for(unsigned int r = 0; r <" << to_string(this->coarsen_factor) << "; r++)" << endl;
            return_code_str << "{" << endl;
            return_code_str << "temp_result[r] =  __shfl_down_sync(0xffffffff, " << result_register_name << "[r], 8);" << endl;
            return_code_str << "}" << endl;
            return_code_str << "if (first_row_thread == temp_row && lane_id < 24){" << endl;
            return_code_str << "for(unsigned int r = 0; r <" << to_string(this->coarsen_factor) << "; r++)" << endl;
            return_code_str << "{" << endl;
            return_code_str << result_register_name << "[r] += temp_result[r];" << endl;
            return_code_str << "}" << endl;
            return_code_str << "}" << endl;
        }
        if (get_config()["VECTOR_WIDTH"].as_float() >= 8)
        {
            return_code_str << "temp_row = __shfl_down_sync(0xffffffff, first_row_thread, 4);" << endl;
            return_code_str << "for(unsigned int r = 0; r <" << to_string(this->coarsen_factor) << "; r++)" << endl;
            return_code_str << "{" << endl;
            return_code_str << "temp_result[r] =  __shfl_down_sync(0xffffffff, " << result_register_name << "[r], 4);" << endl;
            return_code_str << "}" << endl;
            return_code_str << "if (first_row_thread == temp_row && lane_id < 28){" << endl;
            return_code_str << "for(unsigned int r = 0; r <" << to_string(this->coarsen_factor) << "; r++)" << endl;
            return_code_str << "{" << endl;
            return_code_str << result_register_name << "[r] += temp_result[r];" << endl;
            return_code_str << "}" << endl;
            return_code_str << "}" << endl;
        }

        if (get_config()["VECTOR_WIDTH"].as_float() >= 4)
        {
            return_code_str << "temp_row = __shfl_down_sync(0xffffffff, first_row_thread, 2);" << endl;
            return_code_str << "for(unsigned int r = 0; r <" << to_string(this->coarsen_factor) << "; r++)" << endl;
            return_code_str << "{" << endl;
            return_code_str << "temp_result[r] =  __shfl_down_sync(0xffffffff, " << result_register_name << "[r], 2);" << endl;
            return_code_str << "}" << endl;
            return_code_str << "if (first_row_thread == temp_row && lane_id < 30){" << endl;
            return_code_str << "for(unsigned int r = 0; r <" << to_string(this->coarsen_factor) << "; r++)" << endl;
            return_code_str << "{" << endl;
            return_code_str << result_register_name << "[r] += temp_result[r];" << endl;
            return_code_str << "}" << endl;
            return_code_str << "}" << endl;
        }

        if (get_config()["VECTOR_WIDTH"].as_float() >= 2)
        {
            return_code_str << "temp_row = __shfl_down_sync(0xffffffff, first_row_thread, 1);" << endl;
            return_code_str << "for(unsigned int r = 0; r <" << to_string(this->coarsen_factor) << "; r++)" << endl;
            return_code_str << "{" << endl;
            return_code_str << "temp_result[r] =  __shfl_down_sync(0xffffffff, " << result_register_name << "[r], 1);" << endl;
            return_code_str << "}" << endl;
            return_code_str << "if (first_row_thread == temp_row && lane_id < 31){" << endl;
            return_code_str << "for(unsigned int r = 0; r <" << to_string(this->coarsen_factor) << "; r++)" << endl;
            return_code_str << "{" << endl;
            return_code_str << result_register_name << "[r] += temp_result[r];" << endl;
            return_code_str << "}" << endl;
            return_code_str << "}" << endl;
        }
    }
    else
    {
        if (get_config()["VECTOR_WIDTH"].as_float() >= 32)
        {
            return_code_str << "temp_row = __shfl_down_sync(0xffffffff, first_row_thread, 16);" << endl;
            return_code_str << "temp_result[0] =  __shfl_down_sync(0xffffffff, " << result_register_name << "[0], 16);" << endl;
            return_code_str << "if (first_row_thread == temp_row && lane_id < 16){" << endl;
            return_code_str << result_register_name << "[0] += temp_result[0];" << endl;
            return_code_str << "}" << endl;
        }
        if (get_config()["VECTOR_WIDTH"].as_float() >= 16)
        {
            return_code_str << "temp_row = __shfl_down_sync(0xffffffff, first_row_thread, 8);" << endl;
            return_code_str << "temp_result[0] =  __shfl_down_sync(0xffffffff, " << result_register_name << "[0], 8);" << endl;
            return_code_str << "if (first_row_thread == temp_row && lane_id < 24){" << endl;
            return_code_str << result_register_name << "[0] += temp_result[0];" << endl;
        }
        return_code_str << "}" << endl;
        if (get_config()["VECTOR_WIDTH"].as_float() >= 8)
        {
            return_code_str << "temp_row = __shfl_down_sync(0xffffffff, first_row_thread, 4);" << endl;
            return_code_str << "temp_result[0] =  __shfl_down_sync(0xffffffff, " << result_register_name << "[0], 4);" << endl;
            return_code_str << "if (first_row_thread == temp_row && lane_id < 28){" << endl;
            return_code_str << result_register_name << "[0] += temp_result[0];" << endl;
            return_code_str << "}" << endl;
        }
        if (get_config()["VECTOR_WIDTH"].as_float() >= 4)
        {
            return_code_str << "temp_row = __shfl_down_sync(0xffffffff, first_row_thread, 2);" << endl;
            return_code_str << "temp_result[0] =  __shfl_down_sync(0xffffffff, " << result_register_name << "[0], 2);" << endl;
            return_code_str << "if (first_row_thread == temp_row && lane_id < 30){" << endl;
            return_code_str << result_register_name << "[0] += temp_result[0];" << endl;
            return_code_str << "}" << endl;
        }
        if (get_config()["VECTOR_WIDTH"].as_float() >= 2)
        {
            return_code_str << "temp_row = __shfl_down_sync(0xffffffff, first_row_thread, 1);" << endl;
            return_code_str << "temp_result[0] =  __shfl_down_sync(0xffffffff, " << result_register_name << "[0], 1);" << endl;
            return_code_str << "if (first_row_thread == temp_row && lane_id < 31){" << endl;
            return_code_str << result_register_name << "[0] += temp_result[0];" << endl;
            return_code_str << "}" << endl;
        }
    }

    vector<shared_ptr<basic_token>> first_row_indices_of_thread = code_generator_ptr->generate_unfused_memory_access(WARP_META, "first_row_indices", cur_BMW_id_token, true, "first_row_thread");
    string row_name = dynamic_pointer_cast<var_init_token>(first_row_indices_of_thread[0])->get_inited_var_name();
    for (int i = 0; i < first_row_indices_of_thread.size(); i++)
    {
        return_code_str << first_row_indices_of_thread[i]->run() << endl;
    }

    return_code_str << "bool write_flag = " << warp_bit_map->run() << " >> (lane_id) & 1;" << endl;
    if (this->coarsen_factor > 1)
    {
        return_code_str << "if(write_flag == true){" << endl;
        return_code_str << "for(unsigned int q = 0; q < " << to_string(this->coarsen_factor) << "; q++)" << endl;
        return_code_str << "{" << endl;
        return_code_str << "atomicAdd(&y_arr[" << row_name << " * K + dense_matrix_ptr + q] ," << result_register_name << "[q]);" << endl;
        return_code_str << "}" << endl;
        return_code_str << "}" << endl;
    }
    else
    {
        return_code_str << "if(write_flag == true){" << endl;
        return_code_str << "atomicAdd(&y_arr[" << row_name << " * K + dense_matrix_ptr] ," << result_register_name << "[0]);" << endl;
        return_code_str << "}" << endl;
    }

    return_code_str << "}" << endl;
    return_code_str << "else{" << endl;
    // return_code_str << "for (unsigned int d_p = " << dense_ptr;
    // return_code_str << "; d_p < K; d_p +=" << dense_matrix_step->run() << ")" << endl;

    // return_code_str << "{" << endl;

    return_code_str << "if(" << dense_ptr << " >= K)" << endl;
    return_code_str << "return;" << endl;
    
    return_code_str << "unsigned int size_of_buffer = K - " << dense_ptr << ";" << endl;

    if (this->coarsen_factor > 1)
    {
        return_code_str << "temp_row = __shfl_down_sync(0xffffffff, first_row_thread, 16);" << endl;
        return_code_str << "for(unsigned int r = 0; r < size_of_buffer; r++)" << endl;
        return_code_str << "{" << endl;
        return_code_str << "temp_result[r] =  __shfl_down_sync(0xffffffff, " << result_register_name << "[r], 16);" << endl;
        return_code_str << "}" << endl;
        return_code_str << "if (first_row_thread == temp_row && lane_id < 16){" << endl;
        return_code_str << "for(unsigned int r = 0; r < size_of_buffer; r++)" << endl;
        return_code_str << "{" << endl;
        return_code_str << result_register_name << "[r] += temp_result[r];" << endl;
        return_code_str << "}" << endl;
        return_code_str << "}" << endl;

        return_code_str << "temp_row = __shfl_down_sync(0xffffffff, first_row_thread, 8);" << endl;
        return_code_str << "for(unsigned int r = 0; r < size_of_buffer; r++)" << endl;
        return_code_str << "{" << endl;
        return_code_str << "temp_result[r] =  __shfl_down_sync(0xffffffff, " << result_register_name << "[r], 8);" << endl;
        return_code_str << "}" << endl;
        return_code_str << "if (first_row_thread == temp_row && lane_id < 24){" << endl;
        return_code_str << "for(unsigned int r = 0; r < size_of_buffer; r++)" << endl;
        return_code_str << "{" << endl;
        return_code_str << result_register_name << "[r] += temp_result[r];" << endl;
        return_code_str << "}" << endl;
        return_code_str << "}" << endl;

        return_code_str << "temp_row = __shfl_down_sync(0xffffffff, first_row_thread, 4);" << endl;
        return_code_str << "for(unsigned int r = 0; r <size_of_buffer; r++)" << endl;
        return_code_str << "{" << endl;
        return_code_str << "temp_result[r] =  __shfl_down_sync(0xffffffff, " << result_register_name << "[r], 4);" << endl;
        return_code_str << "}" << endl;
        return_code_str << "if (first_row_thread == temp_row && lane_id < 28){" << endl;
        return_code_str << "for(unsigned int r = 0; r < size_of_buffer; r++)" << endl;
        return_code_str << "{" << endl;
        return_code_str << result_register_name << "[r] += temp_result[r];" << endl;
        return_code_str << "}" << endl;
        return_code_str << "}" << endl;

        return_code_str << "temp_row = __shfl_down_sync(0xffffffff, first_row_thread, 2);" << endl;
        return_code_str << "for(unsigned int r = 0; r <size_of_buffer; r++)" << endl;
        return_code_str << "{" << endl;
        return_code_str << "temp_result[r] =  __shfl_down_sync(0xffffffff, " << result_register_name << "[r], 2);" << endl;
        return_code_str << "}" << endl;
        return_code_str << "if (first_row_thread == temp_row && lane_id < 30){" << endl;
        return_code_str << "for(unsigned int r = 0; r < size_of_buffer; r++)" << endl;
        return_code_str << "{" << endl;
        return_code_str << result_register_name << "[r] += temp_result[r];" << endl;
        return_code_str << "}" << endl;
        return_code_str << "}" << endl;

        return_code_str << "temp_row = __shfl_down_sync(0xffffffff, first_row_thread, 1);" << endl;
        return_code_str << "for(unsigned int r = 0; r < size_of_buffer; r++)" << endl;
        return_code_str << "{" << endl;
        return_code_str << "temp_result[r] =  __shfl_down_sync(0xffffffff, " << result_register_name << "[r], 1);" << endl;
        return_code_str << "}" << endl;
        return_code_str << "if (first_row_thread == temp_row && lane_id < 31){" << endl;
        return_code_str << "for(unsigned int r = 0; r <size_of_buffer; r++)" << endl;
        return_code_str << "{" << endl;
        return_code_str << result_register_name << "[r] += temp_result[r];" << endl;
        return_code_str << "}" << endl;
        return_code_str << "}" << endl;
    }
    else
    {
        return_code_str << "temp_row = __shfl_down_sync(0xffffffff, first_row_thread, 16);" << endl;
        return_code_str << "temp_result[0] =  __shfl_down_sync(0xffffffff, " << result_register_name << "[0], 16);" << endl;
        return_code_str << "if (first_row_thread == temp_row && lane_id < 16){" << endl;
        return_code_str << result_register_name << "[0] += temp_result[0];" << endl;
        return_code_str << "}" << endl;

        return_code_str << "temp_row = __shfl_down_sync(0xffffffff, first_row_thread, 8);" << endl;
        return_code_str << "temp_result[0] =  __shfl_down_sync(0xffffffff, " << result_register_name << "[0], 8);" << endl;
        return_code_str << "if (first_row_thread == temp_row && lane_id < 24){" << endl;
        return_code_str << result_register_name << "[0] += temp_result[0];" << endl;
        return_code_str << "}" << endl;

        return_code_str << "temp_row = __shfl_down_sync(0xffffffff, first_row_thread, 4);" << endl;
        return_code_str << "temp_result[0] =  __shfl_down_sync(0xffffffff, " << result_register_name << "[0], 4);" << endl;
        return_code_str << "if (first_row_thread == temp_row && lane_id < 28){" << endl;
        return_code_str << result_register_name << "[0] += temp_result[0];" << endl;
        return_code_str << "}" << endl;

        return_code_str << "temp_row = __shfl_down_sync(0xffffffff, first_row_thread, 2);" << endl;
        return_code_str << "temp_result[0] =  __shfl_down_sync(0xffffffff, " << result_register_name << "[0], 2);" << endl;
        return_code_str << "if (first_row_thread == temp_row && lane_id < 30){" << endl;
        return_code_str << result_register_name << "[0] += temp_result[0];" << endl;
        return_code_str << "}" << endl;

        return_code_str << "temp_row = __shfl_down_sync(0xffffffff, first_row_thread, 1);" << endl;
        return_code_str << "temp_result[0] =  __shfl_down_sync(0xffffffff, " << result_register_name << "[0], 1);" << endl;
        return_code_str << "if (first_row_thread == temp_row && lane_id < 31){" << endl;
        return_code_str << result_register_name << "[0] += temp_result[0];" << endl;
        return_code_str << "}" << endl;
    }

    // vector<shared_ptr<basic_token>> first_row_indices_of_thread = code_generator_ptr->generate_unfused_memory_access(WARP_META, "first_row_indices", cur_BMW_id_token, true, "");
    // string row_name = dynamic_pointer_cast<var_init_token>(first_row_indices_of_thread[0])->get_inited_var_name();
    for (int i = 0; i < first_row_indices_of_thread.size(); i++)
    {
        return_code_str << first_row_indices_of_thread[i]->run() << endl;
    }
    
    return_code_str << "bool write_flag = " << warp_bit_map->run() << " >> (lane_id) & 1;" << endl;

    if (this->coarsen_factor > 1)
    {
        return_code_str << "if(write_flag == true){" << endl;
        return_code_str << "for(unsigned int q = 0; q < size_of_buffer; q++)" << endl;
        return_code_str << "{" << endl;
        return_code_str << "atomicAdd(&y_arr[" << row_name << " * K + dense_matrix_ptr + q] ," << result_register_name << "[q]);" << endl;

        return_code_str << "}" << endl;
        return_code_str << "}" << endl;
    }
    else
    {
        return_code_str << "if(write_flag == true){" << endl;
        return_code_str << "atomicAdd(&y_arr[" << row_name << " * K + dense_matrix_ptr] ," << result_register_name << "[0]);" << endl;
        return_code_str << "}" << endl;
    
    }

    return_code_str << "}" << endl;

    return return_code_str.str();
}

shared_ptr<basic_IO_of_reduction> warp_bit_map_reduce_token::get_output_IO()
{
    return this->output_IO;
}
