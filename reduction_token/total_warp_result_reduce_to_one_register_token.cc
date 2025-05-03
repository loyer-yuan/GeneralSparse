// #include "../reduction_token.hpp"
// #include "../IO_of_reduction.hpp"
// #include "../code_generator.hpp"

// total_warp_result_reduce_to_one_register_token::total_warp_result_reduce_to_one_register_token(shared_ptr<meta_data_set> meta_data_set_ptr, unsigned int coarsen_factor, shared_ptr<code_generator> code_generator_ptr)
//     : reduction_basic_token(WARP_META, meta_data_set_ptr)
// {
//     assert(meta_data_set_ptr != NULL);
//     assert(code_generator_ptr != NULL);
//     assert(code_generator_ptr->check() == true);

//     this->code_generator_ptr = code_generator_ptr;
//     this->coarsen_factor = coarsen_factor;

//     int count = coarsen_factor;

//     shared_ptr<one_register_result_IO_of_reduction> one_register_IO(new one_register_result_IO_of_reduction(WARP_META, count));
//     this->output_IO = one_register_IO;
// }

// // 执行对应的代码生成过程
// string total_warp_result_reduce_to_one_register_token::run()
// {
//     // 将代码生成器复原成智能指针
//     shared_ptr<code_generator> code_generator_ptr = this->code_generator_ptr.lock();

//     if (code_generator_ptr == NULL)
//     {
//         cout << "total_warp_result_reduce_to_one_register_token::run(): the code generator has already destroyed" << endl;
//         assert(false);
//     }

//     assert(code_generator_ptr->check() == true);

//     stringstream return_code_str;

//     string result_register_name = dynamic_pointer_cast<one_register_result_IO_of_reduction>(this->output_IO)->var_name_token_of_IO_register()->run();

//     // 查看当前是不是交错存储，如果没有交错存储，就直接遍历这个BMT的对应的所有数据块

//     // 查看当前所针对的子矩阵
//     int sub_matrix_id = code_generator_ptr->get_sub_matrix_id();

//     assert(sub_matrix_id >= 0);

//     assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_vals", sub_matrix_id) == true);
//     shared_ptr<universal_array> var_array_ptr = this->meta_data_set_ptr
//                                                     ->get_element(GLOBAL_META, "nz_vals", sub_matrix_id)
//                                                     ->get_metadata_arr();

//     data_type data_type_of_result = var_array_ptr->get_data_type();
//     if (data_type_of_result == FLOAT && get_config()["HALF"].as_bool() == true)
//     {
//         data_type_of_result = HALF;
//     }
//     shared_ptr<math_expr_token> dense_matrix_ptr_init = NULL;
//     shared_ptr<math_expr_token> dense_matrix_step = NULL;

//     string dense_ = "((blockIdx.y * blockDim.y) + threadIdx.y) * " + to_string(this->coarsen_factor);
//     dense_matrix_ptr_init = make_shared<math_expr_token>(dense_);
//     dense_matrix_step = make_shared<math_expr_token>("blockDim.y * gridDim.y * " + to_string(this->coarsen_factor));
    

//     // 首先申请一个全局变量作为输出变量
//     shared_ptr<var_name_token> result_register = code_generator_ptr->generate_global_var(data_type_of_result, result_register_name, NULL, this->coarsen_factor);
//     shared_ptr<var_name_token> dense_matrix_ptr = code_generator_ptr->generate_global_var(UNSIGNED_INT, "dense_matrix_ptr", dense_matrix_ptr_init);

//     // 这里开启一个for循环，每个线程遍历对应BMT内所有的nz，将最终的计算内容存到IO的变量中
//     // 获取当前BMT的第一个非零元索引
//     // 查看每一个BMT的首个非零元索引是不是存在
//     // 查看BMT的first_nz_indices是不是存在
//     assert(this->meta_data_set_ptr->is_exist(WARP_META, "first_nz_indices", sub_matrix_id));

//     // 访问的位置
//     shared_ptr<math_expr_token> cur_BMW_id_token(new math_expr_token(code_generator_ptr->code_of_data_block_id_distributed_to_spec_paral(WARP_META)->run()));
//     shared_ptr<math_expr_token> next_BMW_id_token(new math_expr_token(code_generator_ptr->code_of_data_block_id_distributed_to_spec_paral(WARP_META)->run() + "+1"));

//     shared_ptr<basic_token> for_begin_var;
//     shared_ptr<basic_token> for_end_var;
//     string nz_step;
//     string BMW_nz_id = "BMW_nz_id";

//     // 列索引必须存在
//     assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "begin_col_index", sub_matrix_id) == true);

//     // 查看当前子矩阵的首列的列索引
//     unsigned long first_col_index_of_sub_matrix = this->meta_data_set_ptr
//                                                       ->get_element(GLOBAL_META, "begin_col_index", sub_matrix_id)
//                                                       ->get_metadata_arr()
//                                                       ->read_integer_from_arr(0);
//     shared_ptr<math_expr_token> col_and_val_read_index_token_ptr(new math_expr_token(BMW_nz_id));
//     vector<shared_ptr<basic_token>> col_indices_read_token_vec;
//     vector<shared_ptr<basic_token>> vals_read_token_vec;

//     col_indices_read_token_vec = code_generator_ptr->generate_unfused_memory_access(GLOBAL_META, "nz_col_indices", col_and_val_read_index_token_ptr, false);
//     vals_read_token_vec = code_generator_ptr->generate_unfused_memory_access(GLOBAL_META, "nz_vals", col_and_val_read_index_token_ptr, false);

//     for_begin_var = code_generator_ptr->generate_fused_memory_access_with_relative(WARP_META, "first_nz_indices", cur_BMW_id_token);
//     for_end_var = code_generator_ptr->generate_fused_memory_access_with_relative(WARP_META, "first_nz_indices", next_BMW_id_token);
//     nz_step = get_config()["VECTOR_WIDTH"].as_string();

//     string col_indices_read_name = dynamic_pointer_cast<var_init_token>(col_indices_read_token_vec[0])->get_inited_var_name();
//     string var_read_name = dynamic_pointer_cast<var_init_token>(vals_read_token_vec[0])->get_inited_var_name();

//     data_type nz_data_type = this->meta_data_set_ptr->get_element(WARP_META, "first_nz_indices", sub_matrix_id)
//                                  ->get_metadata_arr()
//                                  ->get_data_type();

//     string dense_ptr = dense_matrix_ptr->run();

//     unsigned int r_size = 1;
//     unsigned int a_size = 4;

//     if (this->coarsen_factor == 2)
//     {
//         if (data_type_of_result == FLOAT)
//         {
//             r_size = 1;
//             a_size = 2;
//         }
//         else if (data_type_of_result == DOUBLE)
//         {
//             r_size = 2;
//         }
//     }

//     if (data_type_of_result == HALF)
//     {
//         r_size = 1;
//         a_size = 2;
//     }
//     string vec_acc = "float";
//     if (data_type_of_result == HALF)
//     {
//         vec_acc = "half";
//     }

//     shared_ptr<basic_token> thread_row_indices = code_generator_ptr->generate_fused_memory_access_with_relative(WARP_META, "first_row_indices", cur_BMW_id_token);
//     return_code_str << code_of_data_type(data_type_of_result) << " buffer[" << to_string(this->coarsen_factor) << "];" << endl;
//     if (this->coarsen_factor > 1)
//     {
//         return_code_str << "for (unsigned int p = 0; p <" << to_string(this->coarsen_factor) << "; p++)" << endl;
//         return_code_str << "{" << endl;
//         return_code_str << result_register_name << "[p] = 0;" << endl;
//         return_code_str << "}" << endl;
//     }
//     else
//     {
//         return_code_str << result_register_name << " = 0;" << endl;
//     }

//     return_code_str << "if(" << dense_ptr << " + " << to_string(this->coarsen_factor) << " <= K)";
//     return_code_str << "{" << endl;

//     // return_code_str << "for (unsigned int d_p = " << dense_ptr;
//     // return_code_str << "; d_p < K; d_p +=" << dense_matrix_step->run() << ")" << endl;

//     // return_code_str << "{" << endl;

//     // for循环
//     return_code_str << "for (" << code_of_data_type(nz_data_type) << " " << BMW_nz_id << " = ";
//     return_code_str << for_begin_var->run() << " + threadIdx.x % " + get_config()["VECTOR_WIDTH"].as_string() << "; " << BMW_nz_id << " < " << for_end_var->run() << "; ";
//     return_code_str << BMW_nz_id << "+= " << nz_step << ")" << endl;

//     return_code_str << "{" << endl;

//     for (unsigned int i = 0; i < col_indices_read_token_vec.size(); i++)
//     {
//         return_code_str << col_indices_read_token_vec[i]->run() << endl;
//     }

//     for (unsigned int i = 0; i < vals_read_token_vec.size(); i++)
//     {
//         return_code_str << vals_read_token_vec[i]->run() << endl;
//     }

//     if (this->coarsen_factor > 1)
//     {
//         return_code_str << "for(unsigned int q = 0; q < " << to_string(this->coarsen_factor * r_size / a_size) << "; q++)" << endl;
//         return_code_str << "{" << endl;
//         return_code_str << "*(" << vec_acc << to_string(a_size) << " *)(buffer + q* " << to_string(a_size / r_size) << ") = *("<< vec_acc << to_string(a_size) << " *)";
//         if (first_col_index_of_sub_matrix != 0)
//         {
//             return_code_str << "(x_arr + " << dense_ptr << " + (" << col_indices_read_name << " + " << first_col_index_of_sub_matrix << ") * K + q* " << to_string(a_size / r_size) << ");" << endl;
//         }
//         else
//         {
//             return_code_str << "(x_arr + " << dense_ptr << " + " << col_indices_read_name << " * K + q* " << to_string(a_size / r_size) << ");" << endl;
//         }
//         return_code_str << "}" << endl;

//         return_code_str << "for(unsigned int r = 0; r <" << to_string(this->coarsen_factor) << "; r++)" << endl;
//         return_code_str << "{" << endl;
//         return_code_str << result_register_name << "[r] += " << var_read_name << " * "
//                         << "buffer[r];" << endl;
//         return_code_str << "}" << endl;
//     }
//     else
//     {

//         return_code_str << "buffer[0] = ";
//         if (first_col_index_of_sub_matrix != 0)
//         {
//             return_code_str << "x_arr[" << dense_ptr << " + (" << col_indices_read_name << " + " << first_col_index_of_sub_matrix << ") * K];" << endl;
//         }
//         else
//         {
//             return_code_str << "x_arr[" << dense_ptr << " + " << col_indices_read_name << " * K];" << endl;
//         }

//         return_code_str << result_register_name << " += " << var_read_name << " * buffer[0];" << endl;
//     }

//     // return_code_str << "}" << endl;
//     return_code_str << "}" << endl;

//     if (this->coarsen_factor > 1)
//     {
//         return_code_str << "for(unsigned int r = 0; r <" << to_string(this->coarsen_factor) << "; r++)" << endl;
//         return_code_str << "{" << endl;
//         return_code_str << result_register_name << "[r] += __shfl_down_sync(0xffffffff, " << result_register_name << "[r], 16);" << endl;
//         return_code_str << result_register_name << "[r] += __shfl_down_sync(0xffffffff, " << result_register_name << "[r], 8);" << endl;
//         return_code_str << result_register_name << "[r] += __shfl_down_sync(0xffffffff, " << result_register_name << "[r], 4);" << endl;
//         return_code_str << result_register_name << "[r] += __shfl_down_sync(0xffffffff, " << result_register_name << "[r], 2);" << endl;
//         return_code_str << result_register_name << "[r] += __shfl_down_sync(0xffffffff, " << result_register_name << "[r], 1);" << endl;
//         return_code_str << "}" << endl;
//     }
//     else
//     {
//         return_code_str << result_register_name << " += __shfl_down_sync(0xffffffff, " << result_register_name << ", 16);" << endl;
//         return_code_str << result_register_name << " += __shfl_down_sync(0xffffffff, " << result_register_name << ", 8);" << endl;
//         return_code_str << result_register_name << " += __shfl_down_sync(0xffffffff, " << result_register_name << ", 4);" << endl;
//         return_code_str << result_register_name << " += __shfl_down_sync(0xffffffff, " << result_register_name << ", 2);" << endl;
//         return_code_str << result_register_name << " += __shfl_down_sync(0xffffffff, " << result_register_name << ", 1);" << endl;
//     }

//     vector<shared_ptr<basic_token>> first_row_indices_of_thread = code_generator_ptr->generate_unfused_memory_access(WARP_META, "first_row_indices", cur_BMW_id_token, true, "");
//     string row_name = dynamic_pointer_cast<var_init_token>(first_row_indices_of_thread[0])->get_inited_var_name();
//     for (int i = 0; i < first_row_indices_of_thread.size(); i++)
//     {
//         return_code_str << first_row_indices_of_thread[i]->run() << endl;
//     }

//     if (this->coarsen_factor > 1)
//     {
//         return_code_str << "if(threadIdx.x % 32 == 0){" << endl;
//         return_code_str << "for(unsigned int q = 0; q < " << to_string(this->coarsen_factor * r_size / a_size) << "; q++)" << endl;
//         return_code_str << "{" << endl;
//         return_code_str << "*(" << vec_acc << to_string(a_size) << " *)(y_arr + " << row_name << " * K + dense_matrix_ptr + q * " << to_string(a_size / r_size) << ") = *(" << vec_acc << to_string(a_size) << " *)(" << result_register_name << " + q * " << to_string(a_size / r_size) << ");";
//         return_code_str << "}" << endl;
//         return_code_str << "}" << endl;
//     }
//     else
//     {
//         return_code_str << "if(threadIdx.x % 32 == 0){" << endl;
//         return_code_str << "y_arr[" << row_name << " * K + dense_matrix_ptr] = " << result_register_name << ";";
//         return_code_str << "}" << endl;
//     }

//     return_code_str << "}" << endl;
//     return_code_str << "else{" << endl;
//     // return_code_str << "for (unsigned int d_p = " << dense_ptr;
//     // return_code_str << "; d_p < K; d_p +=" << dense_matrix_step->run() << ")" << endl;

//     // return_code_str << "{" << endl;

//     return_code_str << "if(" << dense_ptr << " >= K)" << endl;
//     return_code_str << "return;" << endl;
    
    
//     return_code_str << "unsigned int size_of_buffer = K - " << dense_ptr << ";" << endl;

//     // for循环
//     return_code_str << "for (" << code_of_data_type(nz_data_type) << " " << BMW_nz_id << " = ";
//     return_code_str << for_begin_var->run() << " + threadIdx.x % " + get_config()["VECTOR_WIDTH"].as_string() << "; " << BMW_nz_id << " < " << for_end_var->run() << "; ";
//     return_code_str << BMW_nz_id << "+= " << nz_step << ")" << endl;

//     return_code_str << "{" << endl;
//     for (unsigned int i = 0; i < col_indices_read_token_vec.size(); i++)
//     {
//         return_code_str << col_indices_read_token_vec[i]->run() << endl;
//     }

//     for (unsigned int i = 0; i < vals_read_token_vec.size(); i++)
//     {
//         return_code_str << vals_read_token_vec[i]->run() << endl;
//     }

//     if (this->coarsen_factor > 1)
//     {
//         return_code_str << "for(unsigned int q = 0; q < size_of_buffer; q++)" << endl;
//         return_code_str << "{" << endl;
//         return_code_str << "buffer[q]"
//                         << " = ";
//         if (first_col_index_of_sub_matrix != 0)
//         {
//             return_code_str << "x_arr[" << dense_ptr << " + (" << col_indices_read_name << " + " << first_col_index_of_sub_matrix << ") * K + q];" << endl;
//         }
//         else
//         {
//             return_code_str << "x_arr[" << dense_ptr << " + " << col_indices_read_name << " * K + q];" << endl;
//         }
//         return_code_str << "}" << endl;
//     }
//     else
//     {
//         return_code_str << "buffer[0] = ";
//         if (first_col_index_of_sub_matrix != 0)
//         {
//             return_code_str << "x_arr[" << dense_ptr << " + (" << col_indices_read_name << " + " << first_col_index_of_sub_matrix << ") * K];" << endl;
//         }
//         else
//         {
//             return_code_str << "x_arr[" << dense_ptr << " + " << col_indices_read_name << " * K];" << endl;
//         }
//     }

//     if (this->coarsen_factor > 1)
//     {
//         return_code_str << "for(unsigned int r = 0; r <" << to_string(this->coarsen_factor) << "; r++)" << endl;
//         return_code_str << "{" << endl;
//         return_code_str << result_register_name << "[r] += " << var_read_name << " * "
//                         << "buffer[r];" << endl;
//         return_code_str << "}" << endl;
//     }
//     else
//     {
//         return_code_str << result_register_name << " += " << var_read_name << " * buffer[0];" << endl;
//     }

//     // return_code_str << "}" << endl;
//     return_code_str << "}" << endl;

//     if (this->coarsen_factor > 1)
//     {
//         return_code_str << "for(unsigned int r = 0; r < size_of_buffer; r++)" << endl;
//         return_code_str << "{" << endl;
//         return_code_str << result_register_name << "[r] += __shfl_down_sync(0xffffffff, " << result_register_name << "[r], 16);" << endl;
//         return_code_str << result_register_name << "[r] += __shfl_down_sync(0xffffffff, " << result_register_name << "[r], 8);" << endl;
//         return_code_str << result_register_name << "[r] += __shfl_down_sync(0xffffffff, " << result_register_name << "[r], 4);" << endl;
//         return_code_str << result_register_name << "[r] += __shfl_down_sync(0xffffffff, " << result_register_name << "[r], 2);" << endl;
//         return_code_str << result_register_name << "[r] += __shfl_down_sync(0xffffffff, " << result_register_name << "[r], 1);" << endl;
//         return_code_str << "}" << endl;
//     }
//     else
//     {
//         return_code_str << result_register_name << " += __shfl_down_sync(0xffffffff, " << result_register_name << ", 16);" << endl;
//         return_code_str << result_register_name << " += __shfl_down_sync(0xffffffff, " << result_register_name << ", 8);" << endl;
//         return_code_str << result_register_name << " += __shfl_down_sync(0xffffffff, " << result_register_name << ", 4);" << endl;
//         return_code_str << result_register_name << " += __shfl_down_sync(0xffffffff, " << result_register_name << ", 2);" << endl;
//         return_code_str << result_register_name << " += __shfl_down_sync(0xffffffff, " << result_register_name << ", 1);" << endl;
//     }

//     // vector<shared_ptr<basic_token>> first_row_indices_of_thread = code_generator_ptr->generate_unfused_memory_access(WARP_META, "first_row_indices", cur_BMW_id_token, true, "");
//     // string row_name = dynamic_pointer_cast<var_init_token>(first_row_indices_of_thread[0])->get_inited_var_name();
//     for (int i = 0; i < first_row_indices_of_thread.size(); i++)
//     {
//         return_code_str << first_row_indices_of_thread[i]->run() << endl;
//     }

//     if(this->coarsen_factor > 1)
//     {
//         return_code_str << "if(threadIdx.x % 32 == 0){" << endl;
//         return_code_str << "for(unsigned int q = 0; q < size_of_buffer; q++)" << endl;
//         return_code_str << "{" << endl;
//         return_code_str << "y_arr[" << row_name << " * K + dense_matrix_ptr + q] = " << result_register_name << "[q];";
//         return_code_str << "}" << endl;
//         return_code_str << "}" << endl;
//     }
//     else
//     {
//         return_code_str << "if(threadIdx.x % 32 == 0){" << endl;
//         return_code_str << "y_arr[" << row_name << " * K + dense_matrix_ptr] = " << result_register_name << ";";
//         return_code_str << "}" << endl;
//     }
//     return_code_str << "}" << endl;

//     return return_code_str.str();
// }

// shared_ptr<basic_IO_of_reduction> total_warp_result_reduce_to_one_register_token::get_output_IO()
// {
//     return this->output_IO;
// }


#include "../reduction_token.hpp"
#include "../IO_of_reduction.hpp"
#include "../code_generator.hpp"

total_warp_result_reduce_to_one_register_token::total_warp_result_reduce_to_one_register_token(shared_ptr<meta_data_set> meta_data_set_ptr, unsigned int coarsen_factor, shared_ptr<code_generator> code_generator_ptr)
    : reduction_basic_token(WARP_META, meta_data_set_ptr)
{
    assert(meta_data_set_ptr != NULL);
    assert(code_generator_ptr != NULL);
    assert(code_generator_ptr->check() == true);

    this->code_generator_ptr = code_generator_ptr;
    this->coarsen_factor = coarsen_factor;

    int count = coarsen_factor;

    shared_ptr<one_register_result_IO_of_reduction> one_register_IO(new one_register_result_IO_of_reduction(WARP_META, count));
    this->output_IO = one_register_IO;
}

// 执行对应的代码生成过程
string total_warp_result_reduce_to_one_register_token::run()
{
    // 将代码生成器复原成智能指针
    shared_ptr<code_generator> code_generator_ptr = this->code_generator_ptr.lock();

    if (code_generator_ptr == NULL)
    {
        cout << "total_warp_result_reduce_to_one_register_token::run(): the code generator has already destroyed" << endl;
        assert(false);
    }

    assert(code_generator_ptr->check() == true);

    stringstream return_code_str;

    string result_register_name = dynamic_pointer_cast<one_register_result_IO_of_reduction>(this->output_IO)->var_name_token_of_IO_register()->run();

    // 查看当前是不是交错存储，如果没有交错存储，就直接遍历这个BMT的对应的所有数据块

    // 查看当前所针对的子矩阵
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
    

    // 首先申请一个全局变量作为输出变量
    shared_ptr<var_name_token> result_register = code_generator_ptr->generate_global_var(data_type_of_result, result_register_name, NULL, this->coarsen_factor);
    shared_ptr<var_name_token> dense_matrix_ptr = code_generator_ptr->generate_global_var(UNSIGNED_INT, "dense_matrix_ptr", dense_matrix_ptr_init);

    // 这里开启一个for循环，每个线程遍历对应BMT内所有的nz，将最终的计算内容存到IO的变量中
    // 获取当前BMT的第一个非零元索引
    // 查看每一个BMT的首个非零元索引是不是存在
    // 查看BMT的first_nz_indices是不是存在
    assert(this->meta_data_set_ptr->is_exist(WARP_META, "first_nz_indices", sub_matrix_id));

    // 访问的位置
    shared_ptr<math_expr_token> cur_BMW_id_token(new math_expr_token(code_generator_ptr->code_of_data_block_id_distributed_to_spec_paral(WARP_META)->run()));
    shared_ptr<math_expr_token> next_BMW_id_token(new math_expr_token(code_generator_ptr->code_of_data_block_id_distributed_to_spec_paral(WARP_META)->run() + "+1"));

    shared_ptr<basic_token> for_begin_var;
    shared_ptr<basic_token> for_end_var;
    string nz_step;
    string BMW_nz_id = "BMW_nz_id";

    // 列索引必须存在
    assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "begin_col_index", sub_matrix_id) == true);

    // 查看当前子矩阵的首列的列索引
    unsigned long first_col_index_of_sub_matrix = this->meta_data_set_ptr
                                                      ->get_element(GLOBAL_META, "begin_col_index", sub_matrix_id)
                                                      ->get_metadata_arr()
                                                      ->read_integer_from_arr(0);
    shared_ptr<math_expr_token> col_and_val_read_index_token_ptr(new math_expr_token(BMW_nz_id));
    vector<shared_ptr<basic_token>> col_indices_read_token_vec;
    vector<shared_ptr<basic_token>> vals_read_token_vec;

    col_indices_read_token_vec = code_generator_ptr->generate_unfused_memory_access(GLOBAL_META, "nz_col_indices", col_and_val_read_index_token_ptr, false);
    vals_read_token_vec = code_generator_ptr->generate_unfused_memory_access(GLOBAL_META, "nz_vals", col_and_val_read_index_token_ptr, false);

    for_begin_var = code_generator_ptr->generate_fused_memory_access_with_relative(WARP_META, "first_nz_indices", cur_BMW_id_token);
    for_end_var = code_generator_ptr->generate_fused_memory_access_with_relative(WARP_META, "first_nz_indices", next_BMW_id_token);
    nz_step = get_config()["VECTOR_WIDTH"].as_string();

    string col_indices_read_name = dynamic_pointer_cast<var_init_token>(col_indices_read_token_vec[0])->get_inited_var_name();
    string var_read_name = dynamic_pointer_cast<var_init_token>(vals_read_token_vec[0])->get_inited_var_name();

    data_type nz_data_type = this->meta_data_set_ptr->get_element(WARP_META, "first_nz_indices", sub_matrix_id)
                                 ->get_metadata_arr()
                                 ->get_data_type();

    string dense_ptr = dense_matrix_ptr->run();

    unsigned int r_size = 1;
    unsigned int a_size = 4;

    if (this->coarsen_factor == 2)
    {
        if (data_type_of_result == FLOAT)
        {
            r_size = 1;
            a_size = 2;
        }
        else if (data_type_of_result == DOUBLE)
        {
            r_size = 2;
        }
    }

    if (data_type_of_result == HALF)
    {
        r_size = 1;
        a_size = 2;
    }
    string vec_acc = "float";
    if (data_type_of_result == HALF)
    {
        vec_acc = "half";
    }

    shared_ptr<basic_token> thread_row_indices = code_generator_ptr->generate_fused_memory_access_with_relative(WARP_META, "first_row_indices", cur_BMW_id_token);
    return_code_str << code_of_data_type(data_type_of_result) << " buffer[" << to_string(this->coarsen_factor) << "];" << endl;
    if (this->coarsen_factor > 1)
    {
        return_code_str << "for (unsigned int p = 0; p <" << to_string(this->coarsen_factor) << "; p++)" << endl;
        return_code_str << "{" << endl;
        return_code_str << result_register_name << "[p] = 0;" << endl;
        return_code_str << "}" << endl;
    }
    else
    {
        return_code_str << result_register_name << "[0] = 0;" << endl;
    }

    return_code_str << "if(" << dense_ptr << " + " << to_string(this->coarsen_factor) << " <= K)";
    return_code_str << "{" << endl;

    // return_code_str << "for (unsigned int d_p = " << dense_ptr;
    // return_code_str << "; d_p < K; d_p +=" << dense_matrix_step->run() << ")" << endl;

    // return_code_str << "{" << endl;

    // for循环
    return_code_str << "for (" << code_of_data_type(nz_data_type) << " " << BMW_nz_id << " = ";
    return_code_str << for_begin_var->run() << " + threadIdx.x % " + get_config()["VECTOR_WIDTH"].as_string() << "; " << BMW_nz_id << " < " << for_end_var->run() << "; ";
    return_code_str << BMW_nz_id << "+= " << nz_step << ")" << endl;

    return_code_str << "{" << endl;

    for (unsigned int i = 0; i < col_indices_read_token_vec.size(); i++)
    {
        return_code_str << col_indices_read_token_vec[i]->run() << endl;
    }

    for (unsigned int i = 0; i < vals_read_token_vec.size(); i++)
    {
        return_code_str << vals_read_token_vec[i]->run() << endl;
    }

    if (this->coarsen_factor > 1)
    {
        return_code_str << "for(unsigned int q = 0; q < " << to_string(this->coarsen_factor * r_size / a_size) << "; q++)" << endl;
        return_code_str << "{" << endl;
        return_code_str << "*(" << vec_acc << to_string(a_size) << " *)(buffer + q* " << to_string(a_size / r_size) << ") = *("<< vec_acc << to_string(a_size) << " *)";
        if (first_col_index_of_sub_matrix != 0)
        {
            return_code_str << "(x_arr + " << dense_ptr << " + (" << col_indices_read_name << " + " << first_col_index_of_sub_matrix << ") * K + q* " << to_string(a_size / r_size) << ");" << endl;
        }
        else
        {
            return_code_str << "(x_arr + " << dense_ptr << " + " << col_indices_read_name << " * K + q* " << to_string(a_size / r_size) << ");" << endl;
        }
        return_code_str << "}" << endl;

        return_code_str << "for(unsigned int r = 0; r <" << to_string(this->coarsen_factor) << "; r++)" << endl;
        return_code_str << "{" << endl;
        return_code_str << result_register_name << "[r] += " << var_read_name << " * "
                        << "buffer[r];" << endl;
        return_code_str << "}" << endl;
    }
    else
    {

        return_code_str << "buffer[0] = ";
        if (first_col_index_of_sub_matrix != 0)
        {
            return_code_str << "x_arr[" << dense_ptr << " + (" << col_indices_read_name << " + " << first_col_index_of_sub_matrix << ") * K];" << endl;
        }
        else
        {
            return_code_str << "x_arr[" << dense_ptr << " + " << col_indices_read_name << " * K];" << endl;
        }

        return_code_str << result_register_name << "[0] += " << var_read_name << " * buffer[0];" << endl;
    }

    // return_code_str << "}" << endl;
    return_code_str << "}" << endl;

    if (this->coarsen_factor > 1)
    {
        return_code_str << "for(unsigned int r = 0; r <" << to_string(this->coarsen_factor) << "; r++)" << endl;
        return_code_str << "{" << endl;
        if (get_config()["VECTOR_WIDTH"].as_float() >= 32)
        {
            return_code_str << result_register_name << "[r] += __shfl_down_sync(0xffffffff, " << result_register_name << "[r], 16);" << endl;
        }

        if (get_config()["VECTOR_WIDTH"].as_float() >= 16)
        {
            return_code_str << result_register_name << "[r] += __shfl_down_sync(0xffffffff, " << result_register_name << "[r], 8);" << endl;
        }

        if (get_config()["VECTOR_WIDTH"].as_float() >= 8)
        {
            return_code_str << result_register_name << "[r] += __shfl_down_sync(0xffffffff, " << result_register_name << "[r], 4);" << endl;
        }

        if (get_config()["VECTOR_WIDTH"].as_float() >= 4)
        {
            return_code_str << result_register_name << "[r] += __shfl_down_sync(0xffffffff, " << result_register_name << "[r], 2);" << endl;
        }

        if (get_config()["VECTOR_WIDTH"].as_float() >= 2)
        {
            return_code_str << result_register_name << "[r] += __shfl_down_sync(0xffffffff, " << result_register_name << "[r], 1);" << endl;
        }
        return_code_str << "}" << endl;
    }
    else
    {
        if (get_config()["VECTOR_WIDTH"].as_float() >= 32)
        {
            return_code_str << result_register_name << "[0] += __shfl_down_sync(0xffffffff, " << result_register_name << "[0], 16);" << endl;
        }

        if (get_config()["VECTOR_WIDTH"].as_float() >= 16)
        {
            return_code_str << result_register_name << "[0] += __shfl_down_sync(0xffffffff, " << result_register_name << "[0], 8);" << endl;
        }

        if (get_config()["VECTOR_WIDTH"].as_float() >= 8)
        {
            return_code_str << result_register_name << "[0] += __shfl_down_sync(0xffffffff, " << result_register_name << "[0], 4);" << endl;
        }

        if (get_config()["VECTOR_WIDTH"].as_float() >= 4)
        {
            return_code_str << result_register_name << "[0] += __shfl_down_sync(0xffffffff, " << result_register_name << "[0], 2);" << endl;
        }

        if (get_config()["VECTOR_WIDTH"].as_float() >= 2)
        {
            return_code_str << result_register_name << "[0] += __shfl_down_sync(0xffffffff, " << result_register_name << "[0], 1);" << endl;
        }
    }

    vector<shared_ptr<basic_token>> first_row_indices_of_thread = code_generator_ptr->generate_unfused_memory_access(WARP_META, "first_row_indices", cur_BMW_id_token, true, "");
    string row_name = dynamic_pointer_cast<var_init_token>(first_row_indices_of_thread[0])->get_inited_var_name();
    for (int i = 0; i < first_row_indices_of_thread.size(); i++)
    {
        return_code_str << first_row_indices_of_thread[i]->run() << endl;
    }

    if (this->coarsen_factor > 1)
    {
        return_code_str << "if(threadIdx.x % " << to_string((int)get_config()["VECTOR_WIDTH"].as_float()) << " == 0){" << endl;
        return_code_str << "for(unsigned int q = 0; q < " << to_string(this->coarsen_factor * r_size / a_size) << "; q++)" << endl;
        return_code_str << "{" << endl;
        return_code_str << "*(" << vec_acc << to_string(a_size) << " *)(y_arr + " << row_name << " * K + dense_matrix_ptr + q * " << to_string(a_size / r_size) << ") = *(" << vec_acc << to_string(a_size) << " *)(" << result_register_name << " + q * " << to_string(a_size / r_size) << ");";
        return_code_str << "}" << endl;
        return_code_str << "}" << endl;
    }
    else
    {
        return_code_str << "if(threadIdx.x % " << to_string((int)get_config()["VECTOR_WIDTH"].as_float()) << " == 0){" << endl;
        return_code_str << "y_arr[" << row_name << " * K + dense_matrix_ptr] = " << result_register_name << "[0];";
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

    // for循环
    return_code_str << "for (" << code_of_data_type(nz_data_type) << " " << BMW_nz_id << " = ";
    return_code_str << for_begin_var->run() << " + threadIdx.x % " + get_config()["VECTOR_WIDTH"].as_string() << "; " << BMW_nz_id << " < " << for_end_var->run() << "; ";
    return_code_str << BMW_nz_id << "+= " << nz_step << ")" << endl;

    return_code_str << "{" << endl;
    for (unsigned int i = 0; i < col_indices_read_token_vec.size(); i++)
    {
        return_code_str << col_indices_read_token_vec[i]->run() << endl;
    }

    for (unsigned int i = 0; i < vals_read_token_vec.size(); i++)
    {
        return_code_str << vals_read_token_vec[i]->run() << endl;
    }

    if (this->coarsen_factor > 1)
    {
        return_code_str << "for(unsigned int q = 0; q < size_of_buffer; q++)" << endl;
        return_code_str << "{" << endl;
        return_code_str << "buffer[q]"
                        << " = ";
        if (first_col_index_of_sub_matrix != 0)
        {
            return_code_str << "x_arr[" << dense_ptr << " + (" << col_indices_read_name << " + " << first_col_index_of_sub_matrix << ") * K + q];" << endl;
        }
        else
        {
            return_code_str << "x_arr[" << dense_ptr << " + " << col_indices_read_name << " * K + q];" << endl;
        }
        return_code_str << "}" << endl;
    }
    else
    {
        return_code_str << "buffer[0] = ";
        if (first_col_index_of_sub_matrix != 0)
        {
            return_code_str << "x_arr[" << dense_ptr << " + (" << col_indices_read_name << " + " << first_col_index_of_sub_matrix << ") * K];" << endl;
        }
        else
        {
            return_code_str << "x_arr[" << dense_ptr << " + " << col_indices_read_name << " * K];" << endl;
        }
    }

    if (this->coarsen_factor > 1)
    {
        return_code_str << "for(unsigned int r = 0; r <" << to_string(this->coarsen_factor) << "; r++)" << endl;
        return_code_str << "{" << endl;
        return_code_str << result_register_name << "[r] += " << var_read_name << " * "
                        << "buffer[r];" << endl;
        return_code_str << "}" << endl;
    }
    else
    {
        return_code_str << result_register_name << "[0] += " << var_read_name << " * buffer[0];" << endl;
    }

    // return_code_str << "}" << endl;
    return_code_str << "}" << endl;

    if (this->coarsen_factor > 1)
    {
        return_code_str << "for(unsigned int r = 0; r < size_of_buffer; r++)" << endl;
        return_code_str << "{" << endl;

        if (get_config()["VECTOR_WIDTH"].as_float() >= 32)
        {
            return_code_str << result_register_name << "[r] += __shfl_down_sync(0xffffffff, " << result_register_name << "[r], 16);" << endl;
        }

        if (get_config()["VECTOR_WIDTH"].as_float() >= 16)
        {
            return_code_str << result_register_name << "[r] += __shfl_down_sync(0xffffffff, " << result_register_name << "[r], 8);" << endl;
        }

        if (get_config()["VECTOR_WIDTH"].as_float() >= 8)
        {
            return_code_str << result_register_name << "[r] += __shfl_down_sync(0xffffffff, " << result_register_name << "[r], 4);" << endl;
        }

        if (get_config()["VECTOR_WIDTH"].as_float() >= 4)
        {
            return_code_str << result_register_name << "[r] += __shfl_down_sync(0xffffffff, " << result_register_name << "[r], 2);" << endl;
        }

        if (get_config()["VECTOR_WIDTH"].as_float() >= 2)
        {
            return_code_str << result_register_name << "[r] += __shfl_down_sync(0xffffffff, " << result_register_name << "[r], 1);" << endl;
        }

        return_code_str << "}" << endl;
    }
    else
    {
        if (get_config()["VECTOR_WIDTH"].as_float() >= 32)
        {
            return_code_str << result_register_name << "[0] += __shfl_down_sync(0xffffffff, " << result_register_name << "[0], 16);" << endl;
        }

        if (get_config()["VECTOR_WIDTH"].as_float() >= 16)
        {
            return_code_str << result_register_name << "[0] += __shfl_down_sync(0xffffffff, " << result_register_name << "[0], 8);" << endl;
        }

        if (get_config()["VECTOR_WIDTH"].as_float() >= 8)
        {
            return_code_str << result_register_name << "[0] += __shfl_down_sync(0xffffffff, " << result_register_name << "[0], 4);" << endl;
        }

        if (get_config()["VECTOR_WIDTH"].as_float() >= 4)
        {
            return_code_str << result_register_name << "[0] += __shfl_down_sync(0xffffffff, " << result_register_name << "[0], 2);" << endl;
        }

        if (get_config()["VECTOR_WIDTH"].as_float() >= 2)
        {
            return_code_str << result_register_name << "[0] += __shfl_down_sync(0xffffffff, " << result_register_name << "[0], 1);" << endl;
        }
    }

    // vector<shared_ptr<basic_token>> first_row_indices_of_thread = code_generator_ptr->generate_unfused_memory_access(WARP_META, "first_row_indices", cur_BMW_id_token, true, "");
    // string row_name = dynamic_pointer_cast<var_init_token>(first_row_indices_of_thread[0])->get_inited_var_name();
    for (int i = 0; i < first_row_indices_of_thread.size(); i++)
    {
        return_code_str << first_row_indices_of_thread[i]->run() << endl;
    }

    if(this->coarsen_factor > 1)
    {
        return_code_str << "if(threadIdx.x % " << to_string((int)get_config()["VECTOR_WIDTH"].as_float()) << " == 0){" << endl;
        return_code_str << "for(unsigned int q = 0; q < size_of_buffer; q++)" << endl;
        return_code_str << "{" << endl;
        return_code_str << "y_arr[" << row_name << " * K + dense_matrix_ptr + q] = " << result_register_name << "[q];";
        return_code_str << "}" << endl;
        return_code_str << "}" << endl;
    }
    else
    {
        return_code_str << "if(threadIdx.x % " << to_string((int)get_config()["VECTOR_WIDTH"].as_float()) << " == 0){" << endl;
        return_code_str << "y_arr[" << row_name << " * K + dense_matrix_ptr] = " << result_register_name << "[0];";
        return_code_str << "}" << endl;
    }
    return_code_str << "}" << endl;

    return return_code_str.str();
}

shared_ptr<basic_IO_of_reduction> total_warp_result_reduce_to_one_register_token::get_output_IO()
{
    return this->output_IO;
}
