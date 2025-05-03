#include "../data_transform_step.hpp"

modify_vals_by_empty_pad_in_submatrix::modify_vals_by_empty_pad_in_submatrix(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id)
    : basic_data_transform_step("modify_vals_by_empty_pad_in_submatrix", meta_data_set_ptr)
{
    assert(target_matrix_id >= 0);
    assert(meta_data_set_ptr != NULL);
    assert(meta_data_set_ptr->check());
    this->target_matrix_id = target_matrix_id;
}

void modify_vals_by_empty_pad_in_submatrix::run(bool check)
{
    if (check)
    {
        // 保证metadata set
        assert(this->meta_data_set_ptr != NULL);
        assert(this->meta_data_set_ptr->check());
        // 保证矩阵号的正确
        assert(this->target_matrix_id >= 0);

        // 存在对应的列索引
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_vals", this->target_matrix_id));
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "begin_row_index", this->target_matrix_id));
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "end_row_index", this->target_matrix_id));
        // 存在对应行索引
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_row_indices", this->target_matrix_id));
    }

    shared_ptr<universal_array> row_indices_ptr = this->meta_data_set_ptr->get_element(GLOBAL_META, "nz_row_indices", this->target_matrix_id)->get_metadata_arr();

    // 查看当前子块的行数量
    unsigned long begin_row_index = this->meta_data_set_ptr->get_element(GLOBAL_META, "begin_row_index", this->target_matrix_id)->get_metadata_arr()->read_integer_from_arr(0);
    unsigned long end_row_index = this->meta_data_set_ptr->get_element(GLOBAL_META, "end_row_index", this->target_matrix_id)->get_metadata_arr()->read_integer_from_arr(0);
    unsigned long real_max_row_index = row_indices_ptr->read_integer_from_arr(row_indices_ptr->get_len() - 1);

    if (check)
    {
        assert(end_row_index >= begin_row_index);
    }

    unsigned long max_logic_relative_row_index = end_row_index - begin_row_index;

    if (real_max_row_index > max_logic_relative_row_index)
    {
        end_row_index = begin_row_index + real_max_row_index;
    }

    unsigned long row_num_of_sub_matrix = end_row_index - begin_row_index + 1;
    // 值索引
    shared_ptr<universal_array> vals_ptr = this->meta_data_set_ptr->get_element(GLOBAL_META, "nz_vals", this->target_matrix_id)->get_metadata_arr();
    data_type cur_float_type = vals_ptr->get_data_type();

    if (check)
    {
        // 当前行索引还没有padding过
        assert(row_indices_ptr->get_len() == vals_ptr->get_len());

        assert(cur_float_type == FLOAT || cur_float_type == DOUBLE);
    }

    vector<double> new_val_indices_vec;

    // 判断是不是pad过空行
    bool is_padded = false;

    // 遍历当前行索引，拷贝到数组中
    for (unsigned long i = 0; i < row_indices_ptr->get_len(); i++)
    {
        unsigned long nz_row_id = row_indices_ptr->read_integer_from_arr(i);
        unsigned long next_nz_row_id = (i == row_indices_ptr->get_len() - 1) ? row_num_of_sub_matrix : row_indices_ptr->read_integer_from_arr(i + 1);
        double nz_val = vals_ptr->read_float_from_arr(i);
        if (next_nz_row_id == nz_row_id)
        {
            new_val_indices_vec.push_back(nz_val);
            continue;
        }
        else if (next_nz_row_id == (nz_row_id + 1))
        {
            new_val_indices_vec.push_back(nz_val);
            continue;
        }
        else
        {
            unsigned long empty_id = nz_row_id;
            while (empty_id < next_nz_row_id)
            {
                new_val_indices_vec.push_back(nz_val);
                nz_val = 0;
                is_padded = true;
                empty_id++;
            }
        }
    }

    if (is_padded == true)
    {
        shared_ptr<data_item_record> begin_row_index_record(new data_item_record(GLOBAL_META, "begin_row_index", this->target_matrix_id));
        this->source_data_item_ptr_vec.push_back(begin_row_index_record);
        shared_ptr<data_item_record> end_row_index_record(new data_item_record(GLOBAL_META, "end_row_index", this->target_matrix_id));
        this->source_data_item_ptr_vec.push_back(end_row_index_record);
        shared_ptr<data_item_record> row_indices_record(new data_item_record(GLOBAL_META, "nz_row_indices", this->target_matrix_id));
        this->source_data_item_ptr_vec.push_back(row_indices_record);
        shared_ptr<data_item_record> val_indices_record(new data_item_record(GLOBAL_META, "nz_vals", this->target_matrix_id));
        this->source_data_item_ptr_vec.push_back(val_indices_record);

        // 删除之前的列索引
        this->meta_data_set_ptr->remove_element(GLOBAL_META, "nz_vals", this->target_matrix_id);

        // 写新的内容到metadata set
        shared_ptr<universal_array> new_val_indices_ptr(new universal_array(&(new_val_indices_vec[0]), new_val_indices_vec.size(), DOUBLE));

        if (cur_float_type == FLOAT)
        {
            new_val_indices_ptr->compress_float_precise();
        }

        shared_ptr<meta_data_item> new_metadata_ptr(new meta_data_item(new_val_indices_ptr, GLOBAL_META, "nz_vals", this->target_matrix_id));
        this->meta_data_set_ptr->add_element(new_metadata_ptr);

        // 增加输出
        shared_ptr<data_item_record> val_indices_record1(new data_item_record(GLOBAL_META, "nz_vals", this->target_matrix_id));
        this->dest_data_item_ptr_vec.push_back(val_indices_record1);
    }

    this->is_run = true;
}

vector<shared_ptr<data_item_record>> modify_vals_by_empty_pad_in_submatrix::get_source_data_item_ptr_in_data_transform_step()
{
    // 执行过
    assert(this->is_run == true);
    // 内容符合要求
    assert(this->target_matrix_id >= 0);

    for (unsigned long i = 0; i < this->source_data_item_ptr_vec.size(); i++)
    {
        assert(this->source_data_item_ptr_vec[i] != NULL);
    }

    return this->source_data_item_ptr_vec;
}

vector<shared_ptr<data_item_record>> modify_vals_by_empty_pad_in_submatrix::get_dest_data_item_ptr_in_data_transform_step_without_check()
{
    // 执行过
    assert(this->is_run == true);
    // 内容符合要求
    assert(this->target_matrix_id >= 0);

    // 其中没有空指针
    for (unsigned long i = 0; i < this->dest_data_item_ptr_vec.size(); i++)
    {
        assert(this->dest_data_item_ptr_vec[i] != NULL);
    }

    return this->dest_data_item_ptr_vec;
}

string modify_vals_by_empty_pad_in_submatrix::convert_to_string()
{
    // 内容符合要求
    assert(this->target_matrix_id >= 0);
    string return_str = "modify_vals_by_empty_pad_in_submatrix::{name:\"" + name + "\",target_matrix_id:" + to_string(target_matrix_id) + "}";
    return return_str;
}