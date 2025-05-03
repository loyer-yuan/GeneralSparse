#include "../data_transform_step.hpp"

modify_col_indices_by_row_pad_in_sub_matrix::modify_col_indices_by_row_pad_in_sub_matrix(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int multiple)
    : basic_data_transform_step("modify_col_indices_by_row_pad_in_sub_matrix", meta_data_set_ptr)
{
    assert(target_matrix_id >= 0);
    assert(meta_data_set_ptr != NULL);
    assert(meta_data_set_ptr->check());
    assert(multiple >= 2);

    this->multiple = multiple;
    this->target_matrix_id = target_matrix_id;
}

void modify_col_indices_by_row_pad_in_sub_matrix::run(bool check)
{
    if (check)
    {
        // 保证metadata set
        assert(this->meta_data_set_ptr != NULL);
        assert(this->meta_data_set_ptr->check());
        // 保证矩阵号的正确
        assert(this->target_matrix_id >= 0);
        assert(this->multiple >= 2);

        // 存在对应的列索引
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_col_indices", this->target_matrix_id));
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "begin_row_index", this->target_matrix_id));
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "end_row_index", this->target_matrix_id));
        // 存在对应行索引
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_row_indices", this->target_matrix_id));
    }

    // 查看当前子块的行数量
    unsigned long begin_row_index = this->meta_data_set_ptr->get_element(GLOBAL_META, "begin_row_index", this->target_matrix_id)->get_metadata_arr()->read_integer_from_arr(0);
    unsigned long end_row_index = this->meta_data_set_ptr->get_element(GLOBAL_META, "end_row_index", this->target_matrix_id)->get_metadata_arr()->read_integer_from_arr(0);
    unsigned long row_num_of_sub_matrix = end_row_index - begin_row_index + 1;

    // 列索引
    shared_ptr<universal_array> col_indices_ptr = this->meta_data_set_ptr->get_element(GLOBAL_META, "nz_col_indices", this->target_matrix_id)->get_metadata_arr();
    // 当前行索引还没有padding过
    shared_ptr<universal_array> row_indices_ptr = NULL;
    if (check)
    {
        // 行索引
        row_indices_ptr = this->meta_data_set_ptr->get_element(GLOBAL_META, "nz_row_indices", this->target_matrix_id)->get_metadata_arr();
        unsigned long last_row_index = row_indices_ptr->read_integer_from_arr(row_indices_ptr->get_len() - 1);

        assert(row_indices_ptr->get_len() == col_indices_ptr->get_len());
        // 之前没有padding过
        assert(last_row_index < row_num_of_sub_matrix);
    }



    // 需要padding，行索引不是对应的倍数
    if (row_num_of_sub_matrix % this->multiple != 0)
    {
        shared_ptr<data_item_record> begin_row_index_record(new data_item_record(GLOBAL_META, "begin_row_index", this->target_matrix_id));
        this->source_data_item_ptr_vec.push_back(begin_row_index_record);
        shared_ptr<data_item_record> end_row_index_record(new data_item_record(GLOBAL_META, "end_row_index", this->target_matrix_id));
        this->source_data_item_ptr_vec.push_back(end_row_index_record);
        shared_ptr<data_item_record> col_indices_record(new data_item_record(GLOBAL_META, "nz_col_indices", this->target_matrix_id));
        this->source_data_item_ptr_vec.push_back(col_indices_record);

        // 目标行数量
        unsigned long new_row_number = (row_num_of_sub_matrix / this->multiple + 1) * this->multiple;

        if (check)
        {
            assert(new_row_number > row_num_of_sub_matrix);
        }
        // 计算要padding的行（非零元）数量
        unsigned long added_row_number = new_row_number - row_num_of_sub_matrix;

        if (check)
        {
            // 两个变量，一个存储当前原始的非零元大小，另一个是当前padding之后的大小

            unsigned long original_nnz = row_indices_ptr->get_len();
            unsigned long nnz_after_padding = original_nnz;
            nnz_after_padding = nnz_after_padding + added_row_number;
            // 检查当前的padding率，如果超过一定数量就要报错，防止潜在的爆内存
            if ((double)nnz_after_padding / (double)original_nnz >= get_config()["PADDING_RATE_UP_BOUND"].as_integer())
            {
                cout << "modify_col_indices_by_row_pad_in_sub_matrix::run(): current padding rate is" << (double)nnz_after_padding / (double)original_nnz << ", higher than" << get_config()["PADDING_RATE_UP_BOUND"].as_integer() << endl;
                assert(false);
            }
        }

        vector<unsigned long> new_col_indices_vec;

        // 遍历当前行索引，拷贝到数组中
        for (unsigned long i = 0; i < col_indices_ptr->get_len(); i++)
        {
            new_col_indices_vec.push_back(col_indices_ptr->read_integer_from_arr(i));
        }

        unsigned long new_col_index = new_col_indices_vec[new_col_indices_vec.size() - 1];

        // 增加新的内容，为了保证局部性，增加的内容为最后一个列索引。
        for (unsigned long i = 1; i <= added_row_number; i++)
        {
            new_col_indices_vec.push_back(new_col_index);
        }

        // 删除之前的列索引
        this->meta_data_set_ptr->remove_element(GLOBAL_META, "nz_col_indices", this->target_matrix_id);

        // 写新的内容到metadata set
        shared_ptr<universal_array> new_col_indices_ptr(new universal_array(&(new_col_indices_vec[0]), new_col_indices_vec.size(), UNSIGNED_LONG));
        shared_ptr<meta_data_item> new_metadata_ptr(new meta_data_item(new_col_indices_ptr, GLOBAL_META, "nz_col_indices", this->target_matrix_id));
        this->meta_data_set_ptr->add_element(new_metadata_ptr);

        // 增加输出
        shared_ptr<data_item_record> col_indices_record1(new data_item_record(GLOBAL_META, "nz_col_indices", this->target_matrix_id));
        this->dest_data_item_ptr_vec.push_back(col_indices_record1);
    }

    this->is_run = true;
}

vector<shared_ptr<data_item_record>> modify_col_indices_by_row_pad_in_sub_matrix::get_source_data_item_ptr_in_data_transform_step()
{
    // 执行过
    assert(this->is_run == true);
    // 内容符合要求
    assert(this->target_matrix_id >= 0);
    assert(this->multiple >= 2);

    // 其中没有空指针
    for (unsigned long i = 0; i < this->source_data_item_ptr_vec.size(); i++)
    {
        assert(this->source_data_item_ptr_vec[i] != NULL);
    }

    return this->source_data_item_ptr_vec;
}

vector<shared_ptr<data_item_record>> modify_col_indices_by_row_pad_in_sub_matrix::get_dest_data_item_ptr_in_data_transform_step_without_check()
{
    // 执行过
    assert(this->is_run == true);
    // 内容符合要求
    assert(this->target_matrix_id >= 0);
    assert(this->multiple >= 2);

    // 其中没有空指针
    for (unsigned long i = 0; i < this->dest_data_item_ptr_vec.size(); i++)
    {
        assert(this->dest_data_item_ptr_vec[i] != NULL);
    }

    return this->dest_data_item_ptr_vec;
}

string modify_col_indices_by_row_pad_in_sub_matrix::convert_to_string()
{
    // 内容符合要求
    assert(this->target_matrix_id >= 0);
    assert(this->multiple >= 2);

    string return_str = "modify_col_indices_by_row_pad_in_sub_matrix::{name:\"" + name + "\",target_matrix_id:" + to_string(target_matrix_id) + ",multiple:" + to_string(multiple) + "}";

    return return_str;
}