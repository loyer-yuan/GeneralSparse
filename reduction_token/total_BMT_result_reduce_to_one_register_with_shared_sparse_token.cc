#include "../reduction_token.hpp"
#include "../IO_of_reduction.hpp"
#include "../code_generator.hpp"

total_BMT_result_reduce_to_one_register_with_shared_sparse_token::total_BMT_result_reduce_to_one_register_with_shared_sparse_token(shared_ptr<meta_data_set> meta_data_set_ptr, unsigned int sparse_coarsen_factor, unsigned int coarsen_factor, shared_ptr<code_generator> code_generator_ptr)
    : reduction_basic_token(THREAD_META, meta_data_set_ptr)
{
    assert(meta_data_set_ptr != NULL);
    assert(code_generator_ptr != NULL);
    assert(code_generator_ptr->check() == true);
    assert(coarsen_factor == 1 || coarsen_factor == 2 || coarsen_factor == 4 || coarsen_factor == 8);

    this->code_generator_ptr = code_generator_ptr;
    this->coarsen_factor = coarsen_factor;
    this->sparse_coarsen_factor = sparse_coarsen_factor;
    int count = coarsen_factor;

    // input IO默认设计空的
    // output IO需要一个对应位置的输出
    shared_ptr<one_register_result_IO_of_reduction> one_register_IO(new one_register_result_IO_of_reduction(THREAD_META, count));
    this->output_IO = one_register_IO;
}

// 执行对应的代码生成过程
string total_BMT_result_reduce_to_one_register_with_shared_sparse_token::run()
{
    // 将代码生成器复原成智能指针
    shared_ptr<code_generator> code_generator_ptr = this->code_generator_ptr.lock();

    if (code_generator_ptr == NULL)
    {
        cout << "total_BMT_result_reduce_to_one_register_with_shared_sparse_token::run(): the code generator has already destroyed" << endl;
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



    data_type nz_data_type = this->meta_data_set_ptr->get_element(THREAD_META, "first_nz_indices", sub_matrix_id)
                                 ->get_metadata_arr()
                                 ->get_data_type();

    data_type nz_data_type_compress = this->meta_data_set_ptr->get_element(GLOBAL_META, "nz_col_indices", sub_matrix_id)
                                 ->get_metadata_arr()
                                 ->get_compress_data_type();


    shared_ptr<math_expr_token> dense_matrix_ptr_init = NULL;
    shared_ptr<math_expr_token> dense_matrix_step = NULL;

    dense_matrix_ptr_init = make_shared<math_expr_token>("(blockIdx.y * blockDim.x) + threadIdx.x * " + to_string(this->coarsen_factor));
    dense_matrix_step = make_shared<math_expr_token>("blockDim.x * gridDim.y * " + to_string(this->coarsen_factor));

    // 首先申请一个全局变量作为输出变量
    shared_ptr<var_name_token> result_register = code_generator_ptr->generate_global_var(data_type_of_result, result_register_name, NULL, this->coarsen_factor);
    shared_ptr<var_name_token> dense_matrix_ptr = code_generator_ptr->generate_global_var(UNSIGNED_INT, "dense_matrix_ptr", dense_matrix_ptr_init);
    shared_ptr<var_name_token> first_row_thread = code_generator_ptr->generate_global_var(UNSIGNED_LONG, "first_row_thread", NULL);
    
    code_generator_ptr->add_new_use_of_shared_mem(data_type_of_result, "input_val", code_generator_ptr->get_block()[1] * code_generator_ptr->get_block()[0] * this->sparse_coarsen_factor);
    code_generator_ptr->add_new_use_of_shared_mem(nz_data_type, "input_col", code_generator_ptr->get_block()[1] * code_generator_ptr->get_block()[0] * this->sparse_coarsen_factor);


    assert(this->meta_data_set_ptr->is_exist(THREAD_META, "first_nz_indices", sub_matrix_id));

    // 访问的位置
    shared_ptr<math_expr_token> cur_BMT_id_token(new math_expr_token(code_generator_ptr->code_of_data_block_id_distributed_to_spec_paral(THREAD_META)->run()));
    shared_ptr<math_expr_token> next_BMT_id_token(new math_expr_token(code_generator_ptr->code_of_data_block_id_distributed_to_spec_paral(THREAD_META)->run() + "+1"));

    shared_ptr<basic_token> for_begin_var;
    shared_ptr<basic_token> for_end_var;
    string nz_step;
    string BMT_nz_id = "BMT_nz_id";

    // 列索引必须存在
    assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "begin_col_index", sub_matrix_id) == true);

    // 查看当前子矩阵的首列的列索引
    unsigned long first_col_index_of_sub_matrix = this->meta_data_set_ptr
                                                      ->get_element(GLOBAL_META, "begin_col_index", sub_matrix_id)
                                                      ->get_metadata_arr()
                                                      ->read_integer_from_arr(0);

    for_begin_var = code_generator_ptr->generate_fused_memory_access_with_relative(THREAD_META, "first_nz_indices", cur_BMT_id_token);
    for_end_var = code_generator_ptr->generate_fused_memory_access_with_relative(THREAD_META, "first_nz_indices", next_BMT_id_token);
    nz_step = "blockDim.x * " + to_string(this->sparse_coarsen_factor);


    shared_ptr<math_expr_token> col_and_val_read_index_token_ptr(new math_expr_token(BMT_nz_id));
    vector<shared_ptr<basic_token>> col_indices_read_token_vec;
    vector<shared_ptr<basic_token>> vals_read_token_vec;


    col_indices_read_token_vec = code_generator_ptr->generate_unfused_memory_access(GLOBAL_META, "nz_col_indices", col_and_val_read_index_token_ptr, false);
    vals_read_token_vec = code_generator_ptr->generate_unfused_memory_access(GLOBAL_META, "nz_vals", col_and_val_read_index_token_ptr, false);


    string col_indices_read_name = dynamic_pointer_cast<var_init_token>(col_indices_read_token_vec[0])->get_inited_var_name();
    string var_read_name = dynamic_pointer_cast<var_init_token>(vals_read_token_vec[0])->get_inited_var_name();

    string dense_ptr = dense_matrix_ptr->run();

    if (this->coarsen_factor > 1)
    {
        return_code_str << "for (unsigned int p = 0; p <" << to_string(this->coarsen_factor) << "; p++)" << endl;
        return_code_str << "{" << endl;
        return_code_str << result_register_name << "[p] = 0;" << endl;
        return_code_str << "}" << endl;
    }
    else
    {
        return_code_str << result_register_name << " = 0;" << endl;
    }

    return_code_str << "for (unsigned int p = 0; p < blockDim.x * blockDim.y; p++)" << endl;
    return_code_str << "{" << endl;
    return_code_str << "input_col[p] = 0;" << endl;
    return_code_str << "}" << endl;

    return_code_str << "for (unsigned int p = 0; p < blockDim.x * blockDim.y; p++)" << endl;
    return_code_str << "{" << endl;
    return_code_str << "input_val[p] = 0;" << endl;
    return_code_str << "}" << endl;
    return_code_str << code_of_data_type(nz_data_type) << " " << BMT_nz_id << " = " << for_begin_var->run() << " + " << this->sparse_coarsen_factor << " * threadIdx.x" << ";" << endl;
    
    
    return_code_str << "for (; " << BMT_nz_id << " < " << for_end_var->run() << "; ";
    return_code_str << BMT_nz_id << "+= " << nz_step << ")" << endl;

    return_code_str << "{" << endl;

    if (this->sparse_coarsen_factor == 1)
    {
        for (unsigned int i = 0; i < col_indices_read_token_vec.size(); i++)
        {
            return_code_str << col_indices_read_token_vec[i]->run() << endl;
        }

        for (unsigned int i = 0; i < vals_read_token_vec.size(); i++)
        {
            return_code_str << vals_read_token_vec[i]->run() << endl;
        }

        return_code_str << "input_col[threadIdx.y * blockDim.x + threadIdx.x] = " << col_indices_read_name << ";" << endl;

        return_code_str << "input_val[threadIdx.y * blockDim.x + threadIdx.x] = " << var_read_name << ";" << endl;
    }
    else
    {
        return_code_str << code_of_data_type(nz_data_type_compress) << " " << col_indices_read_token_vec[0]->get_inited_var_name() << "["<< this->sparse_coarsen_factor << "];" << endl;
        return_code_str << code_of_data_type(data_type_of_result) << " " << vals_read_token_vec[0]->get_inited_var_name() << "["<< this->sparse_coarsen_factor << "];" << endl;
        string col_vec_name = get_metadata_item_name(GLOBAL_META, "nz_col_indices", code_generator_ptr->get_sub_matrix_id());
        string val_vec_name = get_metadata_item_name(GLOBAL_META, "nz_vals", code_generator_ptr->get_sub_matrix_id());

        if (nz_data_type_compress == UNSIGNED_CHAR && this->sparse_coarsen_factor == 2)
        {
            return_code_str << "*(unsigned short *)" << col_indices_read_name << " = *(unsigned short *)(" << col_vec_name << " + " << BMT_nz_id << ");" << endl;
        }
        else if (nz_data_type_compress == UNSIGNED_CHAR && this->sparse_coarsen_factor == 4)
        {
            return_code_str << "*(unsigned int *)" << col_indices_read_name << " = *(unsigned int *)(" << col_vec_name << " + " << BMT_nz_id << ");" << endl;

        }
        else if (nz_data_type_compress == UNSIGNED_SHORT && this->sparse_coarsen_factor == 2)
        {
            return_code_str << "*(unsigned int *)" << col_indices_read_name << " = *(unsigned int *)(" << col_vec_name << " + " << BMT_nz_id << ");" << endl;
        }
        else if (nz_data_type_compress == UNSIGNED_SHORT && this->sparse_coarsen_factor == 4)
        {
            return_code_str << "*(unsigned long *)" << col_indices_read_name << " = *(unsigned long *)(" << col_vec_name << " + " << BMT_nz_id << ");" << endl;

        }
        else if (nz_data_type_compress == UNSIGNED_INT && this->sparse_coarsen_factor == 2)
        {
            return_code_str << "*(unsigned long *)" << col_indices_read_name << " = *(unsigned long *)(" << col_vec_name << " + " << BMT_nz_id << ");" << endl;
        }
        else if (nz_data_type_compress == UNSIGNED_INT && this->sparse_coarsen_factor == 4)
        {
            assert(false);
        }
        else if (nz_data_type_compress == UNSIGNED_LONG)
        {
            assert(false);
        }

        if(data_type_of_result == HALF && this->sparse_coarsen_factor == 2)
        {
            return_code_str << "*(half2 *)" << var_read_name << " = *(half2 *)(" << val_vec_name << " + " << BMT_nz_id << ");" << endl;
        }else if(data_type_of_result == HALF && this->sparse_coarsen_factor == 4)
        {
            return_code_str << "*(float2 *)" << var_read_name << " = *(float2 *)(" << val_vec_name << " + " << BMT_nz_id << ");" << endl;
        }else if(data_type_of_result == FLOAT && this->sparse_coarsen_factor == 2)
        {
            return_code_str << "*(float2 *)" << var_read_name << " = *(float2 *)(" << val_vec_name << " + " << BMT_nz_id << ");" << endl;
        }else if(data_type_of_result == FLOAT && this->sparse_coarsen_factor == 4)
        {
            return_code_str << "*(float4 *)" << var_read_name << " = *(float4 *)(" << val_vec_name << " + " << BMT_nz_id << ");" << endl;
        }else if(data_type_of_result == DOUBLE && this->sparse_coarsen_factor == 2)
        {
            return_code_str << "*(float4 *)" << var_read_name << " = *(float4 *)(" << val_vec_name << " + " << BMT_nz_id << ");" << endl;
        }
        else if(data_type_of_result == DOUBLE && this->sparse_coarsen_factor == 4)
        {
            assert(false);
        }


        return_code_str << "for(int iter = 0; iter < " << this->sparse_coarsen_factor << "; iter++)" << endl;
        return_code_str << "{" << endl;
        return_code_str << "input_col[(threadIdx.y * blockDim.x + threadIdx.x) * " << this->sparse_coarsen_factor << " + iter] = " << col_indices_read_name << "[iter];" << endl;
        return_code_str << "input_val[(threadIdx.y * blockDim.x + threadIdx.x) * " << this->sparse_coarsen_factor << " + iter] = " << var_read_name << "[iter];" << endl;
        return_code_str << "}" << endl;


    }

    return_code_str << "__syncthreads();" << endl;

    return_code_str << code_of_data_type(data_type_of_result) << " input_dense[" << to_string(this->coarsen_factor) << "];" << endl;

 



    return_code_str << "for(int i = 0; i < blockDim.x * " << this->sparse_coarsen_factor << "; i++)" << endl;
    return_code_str << "{" << endl;

    if (this->coarsen_factor == 1)
    {
        return_code_str << "input_dense[0] = " << endl;
        if (first_col_index_of_sub_matrix != 0)
        {
            return_code_str << "__ldg(&(x_arr[dense_matrix_ptr + (input_col[threadIdx.y * blockDim.x + i] + " << first_col_index_of_sub_matrix << ") * K]));" << endl;
        }
        else
        {
            return_code_str << "__ldg(&(x_arr[dense_matrix_ptr + input_col[threadIdx.y * blockDim.x + i] * K]));" << endl;
        }
    }
    else
    {
        if (this->coarsen_factor == 4 && data_type_of_result == HALF)
        {
            return_code_str << "*(" << code_of_data_type(FLOAT) << to_string(2) << " *)" << "(input_dense) = " << "*(" << code_of_data_type(FLOAT) << to_string(2) << " *)";
        }
        else
        {
            return_code_str << "*(" << code_of_data_type(data_type_of_result) << to_string(coarsen_factor) << " *)" << "(input_dense) = " << "*(" << code_of_data_type(data_type_of_result) << to_string(coarsen_factor) << " *)";
        }

        if (first_col_index_of_sub_matrix != 0)
        {
            return_code_str << "(x_arr + " << dense_ptr << " + (" << "input_col[threadIdx.y * blockDim.x + i]" << " + " << first_col_index_of_sub_matrix << ") * K);" << endl;
        }
        else
        {
            return_code_str << "(x_arr +" << dense_ptr << " + " << "input_col[threadIdx.y * blockDim.x + i] * K" << ");" << endl;
        }
    }

    if (this->coarsen_factor > 1)
    {
        return_code_str << "for(unsigned int r = 0; r <" << to_string(this->coarsen_factor) << "; r++)" << endl;
        return_code_str << "{" << endl;
        return_code_str << result_register_name << "[r] +=  input_val[threadIdx.y * blockDim.x + i] * "
                        << "input_dense[r];" << endl;
        return_code_str << "}" << endl;
    }
    else
    {
        return_code_str << result_register_name << " +=  input_val[threadIdx.y * blockDim.x + i] * "
                        << "input_dense[0];" << endl;
    }
    return_code_str << "}" << endl;


    // return_code_str << "}" << endl;
    return_code_str << "}" << endl;

    shared_ptr<basic_token> first_row_indices_of_thread_sorted = code_generator_ptr->generate_fused_memory_access_with_relative(THREAD_META, "first_row_indices", cur_BMT_id_token);
    return_code_str << first_row_thread->run() << " = " <<  first_row_indices_of_thread_sorted->run() << ";" << endl;
    vector<shared_ptr<basic_token>> first_row_indices_of_thread = code_generator_ptr->generate_unfused_memory_access(THREAD_META, "first_row_indices", cur_BMT_id_token, true, "first_row_thread");
    for (int i = 0; i < first_row_indices_of_thread.size(); i++)
    {
            return_code_str << first_row_indices_of_thread[i]->run() << endl;
    }
    string row_name = dynamic_pointer_cast<var_init_token>(first_row_indices_of_thread[0])->get_inited_var_name();

    if (this->coarsen_factor > 1)
    {
        return_code_str << "for(unsigned int r = 0; r <" << to_string(this->coarsen_factor) << "; r++)" << endl;
        return_code_str << "{" << endl;
        return_code_str << "y_arr[" << row_name << " * K + dense_matrix_ptr + r] = " << result_register_name << "[r];" << endl;
        return_code_str << "}" << endl;
    }
    else
    {
        return_code_str << "y_arr[" << row_name << " * K + dense_matrix_ptr] = " << result_register_name << ";" << endl;
    }


    if (code_generator_ptr->reduction_token_is_existing(TBLOCK_META))
    {
        shared_ptr<math_expr_token> zero_(new math_expr_token("0"));
        shared_ptr<var_name_token> segment_offset = code_generator_ptr->generate_fused_memory_access(THREAD_META, "segment_offset", cur_BMT_id_token);
        shared_ptr<var_name_token> segment_offset_thread = code_generator_ptr->generate_global_var(UNSIGNED_LONG, "segment_offset_thread", zero_);
        return_code_str << segment_offset_thread->run() << " = " << segment_offset->run() << ";" << endl;

        shared_ptr<var_name_token> tblock_bit_map = code_generator_ptr->generate_fused_memory_access(THREAD_META, "bit_map_of_thread", cur_BMT_id_token);
        shared_ptr<var_name_token> tblock_bit_map_thread = code_generator_ptr->generate_global_var(UNSIGNED_LONG, "bit_map_of_thread", NULL);
        return_code_str << tblock_bit_map_thread->run() << " = " << tblock_bit_map->run() << ";" << endl;
    }

    this->is_terminal = true;

    return return_code_str.str();
}

shared_ptr<basic_IO_of_reduction>total_BMT_result_reduce_to_one_register_with_shared_sparse_token::get_output_IO()
{
    return this->output_IO;
}
