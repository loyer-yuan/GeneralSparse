#include "../reduction_token.hpp"
#include "../IO_of_reduction.hpp"
#include "../code_generator.hpp"

tblock_bit_map_reduce_token::tblock_bit_map_reduce_token(shared_ptr<meta_data_set> meta_data_set_ptr, unsigned int coarsen_factor, shared_ptr<code_generator> code_generator_ptr)
    : reduction_basic_token(TBLOCK_META, meta_data_set_ptr)
{
    assert(meta_data_set_ptr != NULL);
    assert(code_generator_ptr != NULL);
    assert(code_generator_ptr->check() == true);

    this->code_generator_ptr = code_generator_ptr;
    this->coarsen_factor = coarsen_factor;

    int count = coarsen_factor;

     shared_ptr<one_register_result_IO_of_reduction> one_register_IO_in(new one_register_result_IO_of_reduction(THREAD_META, count));
    this->input_IO = one_register_IO_in;

    shared_ptr<one_register_result_IO_of_reduction> one_register_IO(new one_register_result_IO_of_reduction(TBLOCK_META, count));
    this->output_IO = one_register_IO;
}

// 执行对应的代码生成过程
string tblock_bit_map_reduce_token::run()
{
    // 将代码生成器复原成智能指针
    shared_ptr<code_generator> code_generator_ptr = this->code_generator_ptr.lock();

    if (code_generator_ptr == NULL)
    {
        cout << "tblock_bit_map_reduce_token::run(): the code generator has already destroyed" << endl;
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
    
    dense_matrix_ptr_init = make_shared<math_expr_token>("(blockIdx.y * blockDim.x) + threadIdx.x");
    dense_matrix_step = make_shared<math_expr_token>("blockDim.x * gridDim.y * " + to_string(this->coarsen_factor));

    // 首先申请一个全局变量作为输出变量
    shared_ptr<var_name_token> result_register = code_generator_ptr->generate_global_var(data_type_of_result, result_register_name, NULL, this->coarsen_factor);
    shared_ptr<var_name_token> dense_matrix_ptr = code_generator_ptr->generate_global_var(UNSIGNED_INT, "dense_matrix_ptr", dense_matrix_ptr_init);
    shared_ptr<var_name_token> first_row_thread = code_generator_ptr->generate_global_var(UNSIGNED_LONG, "first_row_thread", NULL);

    // 这里开启一个for循环，每个线程遍历对应BMT内所有的nz，将最终的计算内容存到IO的变量中
    // 获取当前BMT的第一个非零元索引
    // 查看每一个BMT的首个非零元索引是不是存在
    // 查看BMT的first_nz_indices是不是存在
    assert(this->meta_data_set_ptr->is_exist(TBLOCK_META, "first_nz_indices", sub_matrix_id));

    // 访问的位置
    shared_ptr<math_expr_token> cur_BMT_id_token(new math_expr_token(code_generator_ptr->code_of_data_block_id_distributed_to_spec_paral(THREAD_META)->run()));
    shared_ptr<math_expr_token> cur_BMTB_id_token(new math_expr_token(code_generator_ptr->code_of_data_block_id_distributed_to_spec_paral(TBLOCK_META)->run()));
    shared_ptr<math_expr_token> next_BMTB_id_token(new math_expr_token(code_generator_ptr->code_of_data_block_id_distributed_to_spec_paral(TBLOCK_META)->run() + "+1"));




    code_generator_ptr->add_new_use_of_shared_mem(data_type_of_result, "shared_result", code_generator_ptr->get_block()[1] * code_generator_ptr->get_block()[0] * this->coarsen_factor);
    code_generator_ptr->add_new_use_of_shared_mem(data_type_of_result, "shared_result_2", code_generator_ptr->get_block()[1] * code_generator_ptr->get_block()[0] * this->coarsen_factor);

    data_type nz_data_type = this->meta_data_set_ptr->get_element(TBLOCK_META, "first_nz_indices", sub_matrix_id)
                                 ->get_metadata_arr()
                                 ->get_data_type();

    string dense_ptr = dense_matrix_ptr->run();

    shared_ptr<var_name_token> tblock_first_BMT = code_generator_ptr->generate_fused_memory_access_with_relative(TBLOCK_META, "first_BMT_indices", cur_BMTB_id_token);
    shared_ptr<var_name_token> tblock_first_BMT_next = code_generator_ptr->generate_fused_memory_access_with_relative(TBLOCK_META, "first_BMT_indices", next_BMTB_id_token);

    if (this->coarsen_factor > 1)
    {
        return_code_str << "if(" << code_generator_ptr->get_for_loop_begin_ptr(THREAD_META)->run() << " + " << tblock_first_BMT->run() << " >= " << tblock_first_BMT_next->run() << "){" << endl;

        return_code_str << "for (int i = 0; i < " << this->coarsen_factor << "; i ++ ){" << endl;
        return_code_str << input_result_name << "[i] = 0;" << endl;
        return_code_str << "}" << endl;
        return_code_str << "}" << endl;

        return_code_str << "for (int i = 0; i < " << this->coarsen_factor << "; i ++ ){" << endl;

        return_code_str << "shared_result[blockDim.x * i + threadIdx.x + blockDim.x * threadIdx.y * " << this->coarsen_factor << " ] = " << input_result_name << "[i];" << endl;
        return_code_str << "shared_result_2[blockDim.x * i + threadIdx.x + blockDim.x * threadIdx.y * " << this->coarsen_factor << " ] = 0;" << endl;

        return_code_str << "}" << endl;

        return_code_str << "__syncthreads();" << endl;

        return_code_str << "for(int j = threadIdx.y; j >= 0; j--){" << endl;
        
        return_code_str << "if(threadIdx.y >= j){" << endl;

        return_code_str << "for (int i = 0; i < " << this->coarsen_factor << "; i++ ){" << endl;

        return_code_str << "shared_result_2[blockDim.x * i + threadIdx.x + blockDim.x * threadIdx.y * " << this->coarsen_factor << " ] += " << "shared_result[blockDim.x * i + threadIdx.x + blockDim.x * threadIdx.y * " << this->coarsen_factor << " - j * blockDim.x * " << this->coarsen_factor << "];" << endl;

        return_code_str << "}" << endl;

        return_code_str << "}" << endl;

        return_code_str << "}" << endl;

        return_code_str << "__syncthreads();" << endl;

        return_code_str << "for (int i = 0; i < " << this->coarsen_factor << "; i++ ){" << endl;

        return_code_str << result_register_name << "[i] = " << "shared_result_2[blockDim.x * i + threadIdx.x + blockDim.x * threadIdx.y * " << this->coarsen_factor << " + segment_offset_thread * blockDim.x * " << this->coarsen_factor << "] - " << "shared_result_2[blockDim.x * i + threadIdx.x + blockDim.x * threadIdx.y * " << this->coarsen_factor << " ] + " << input_result_name << "[i];" << endl;

        return_code_str << "}" << endl;

        return_code_str << "__syncthreads();" << endl;

        // vector<shared_ptr<basic_token>> first_row_indices_of_thread = code_generator_ptr->generate_unfused_memory_access(THREAD_META, "first_row_indices", cur_BMT_id_token, true, "");
        // string row_name = dynamic_pointer_cast<var_init_token>(first_row_indices_of_thread[0])->get_inited_var_name();
        // for (int i = 0; i < first_row_indices_of_thread.size(); i++)
        // {
        //     return_code_str << first_row_indices_of_thread[i]->run() << endl;
        // }

        return_code_str << "if(bit_map_of_thread == true){" << endl;

        return_code_str << "for(int factor = 0; factor <" + to_string(this->coarsen_factor) + ";factor++){" << endl;
        return_code_str << "atomicAdd(&y_arr[first_row_thread * K + dense_matrix_ptr + factor], " << result_register_name << "[factor]);" << endl;
        return_code_str << "}" << endl;

        return_code_str << "}" << endl;
    }
    else
    {

        return_code_str << "if(" << code_generator_ptr->get_for_loop_begin_ptr(THREAD_META)->run() << " + " << tblock_first_BMT->run() << " >= " << tblock_first_BMT_next->run() << "){" << endl;
        return_code_str << input_result_name << "[0] = 0;" << endl;
        return_code_str << "}" << endl;

        return_code_str << "shared_result[threadIdx.x + blockDim.x * threadIdx.y] = " << input_result_name << "[0];" << endl;
        return_code_str << "shared_result_2[threadIdx.x + blockDim.x * threadIdx.y] = 0;" << endl;

        return_code_str << "__syncthreads();" << endl;

        return_code_str << "for(int j = threadIdx.y; j >= 0; j--){" << endl;

        return_code_str << "if(threadIdx.y >= j){" << endl;

        return_code_str << "shared_result_2[threadIdx.x + blockDim.x * threadIdx.y] += " << "shared_result[threadIdx.x + blockDim.x * threadIdx.y - j * blockDim.x];" << endl;

        return_code_str << "}" << endl;

        return_code_str << "}" << endl;

        return_code_str << "__syncthreads();" << endl;


        return_code_str << result_register_name << "[0] = " << "shared_result_2[threadIdx.x + blockDim.x * threadIdx.y + segment_offset_thread * blockDim.x] - " << "shared_result_2[threadIdx.x + blockDim.x * threadIdx.y] + " << input_result_name << "[0];" << endl;

        return_code_str << "__syncthreads();" << endl;

        // vector<shared_ptr<basic_token>> first_row_indices_of_tblock = code_generator_ptr->generate_unfused_memory_access(TBLOCK_META, "first_row_indices", cur_BMTB_id_token, true, "");
        // string row_name = dynamic_pointer_cast<var_init_token>(first_row_indices_of_tblock[0])->get_inited_var_name();
        // for (int i = 0; i < first_row_indices_of_tblock.size(); i++)
        // {
        //     return_code_str << first_row_indices_of_tblock[i]->run() << endl;
        // }

        return_code_str << "if(bit_map_of_thread == true){" << endl;
        return_code_str << "atomicAdd(&y_arr[first_row_thread * K + dense_matrix_ptr], " << result_register_name << "[0]);" << endl;
        return_code_str << "}" << endl;

    }
    return return_code_str.str();
}

shared_ptr<basic_IO_of_reduction> tblock_bit_map_reduce_token::get_output_IO()
{
    return this->output_IO;
}

