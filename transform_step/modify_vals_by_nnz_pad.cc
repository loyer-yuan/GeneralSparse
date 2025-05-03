#include "../data_transform_step.hpp"

modify_vals_by_nnz_pad::modify_vals_by_nnz_pad(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int nnz_target)
    : basic_data_transform_step("modify_vals_by_nnz_pad", meta_data_set_ptr)
{
    assert(target_matrix_id >= 0);
    assert(meta_data_set_ptr != NULL);
    assert(meta_data_set_ptr->check());
    this->target_matrix_id = target_matrix_id;
    this->nnz_target = nnz_target;
}

void modify_vals_by_nnz_pad::run(bool check)
{
    if (check)
    {
        // 保证metadata set
        assert(this->meta_data_set_ptr != NULL);
        assert(this->meta_data_set_ptr->check());
        // 保证矩阵号的正确
        assert(this->target_matrix_id >= 0);
        assert(this->nnz_target > 0);

        // 存在对应索引
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_vals", this->target_matrix_id));
    }

    // 行索引
    shared_ptr<universal_array> vals_ptr = this->meta_data_set_ptr->get_element(GLOBAL_META, "nz_vals", this->target_matrix_id)->get_metadata_arr();

    vector<double> new_vals_vec;
    data_type cur_data_type = vals_ptr->get_data_type();

    // 判断是不是pad过
    bool is_padded = false;
    unsigned long nz_num = vals_ptr->get_len();
    if (nz_num % this->nnz_target != 0)
    {
        unsigned long new_nz_number = (nz_num / this->nnz_target + 1) * this->nnz_target;
        unsigned long added_nz_number = new_nz_number - nz_num;

        if (check)
        {
            // 检查当前的padding率，如果超过一定数量就要报错，防止潜在的爆内存
            if ((double)new_nz_number / (double)nz_num >= get_config()["PADDING_RATE_UP_BOUND"].as_integer())
            {
                cout << "modify_vals_by_nnz_pad::run(): current padding rate is" << (double)new_nz_number / (double)nz_num << ", higher than" << get_config()["PADDING_RATE_UP_BOUND"].as_integer() << endl;
                assert(false);
            }
        }

        for (unsigned long i = 0; i < vals_ptr->get_len(); i++)
        {
            new_vals_vec.push_back(vals_ptr->read_float_from_arr(i));
        }

        for (unsigned long i = 0; i < added_nz_number; i++)
        {
            new_vals_vec.push_back(0);
        }

        is_padded = true;

    }

    if (is_padded == true)
    {
        shared_ptr<data_item_record> vals_record(new data_item_record(GLOBAL_META, "nz_vals", this->target_matrix_id));
        this->source_data_item_ptr_vec.push_back(vals_record);

        // 删除之前的行索引
        this->meta_data_set_ptr->remove_element(GLOBAL_META, "nz_vals", this->target_matrix_id);

        // 写新的内容到metadata set
        shared_ptr<universal_array> new_vals_ptr(new universal_array(&(new_vals_vec[0]), new_vals_vec.size(), DOUBLE));

        // 如果是单精度，就需要压缩精度
        if (cur_data_type == FLOAT)
        {
            new_vals_ptr->compress_float_precise();
            if (check)
            {
                assert(new_vals_ptr->get_data_type() == FLOAT);
            }
        }
        // 写新的内容到metadata set
        shared_ptr<meta_data_item> new_metadata_ptr(new meta_data_item(new_vals_ptr, GLOBAL_META, "nz_vals", this->target_matrix_id));
        this->meta_data_set_ptr->add_element(new_metadata_ptr);

        // 增加输出
        shared_ptr<data_item_record> nz_vals_record(new data_item_record(GLOBAL_META, "nz_vals", this->target_matrix_id));
        this->dest_data_item_ptr_vec.push_back(nz_vals_record);
    }

    this->is_run = true;
}

vector<shared_ptr<data_item_record>> modify_vals_by_nnz_pad::get_source_data_item_ptr_in_data_transform_step()
{
    // 执行过
    assert(this->is_run == true);
    // 内容符合要求
    assert(this->target_matrix_id >= 0);
    assert(this->nnz_target > 0);

    for (unsigned long i = 0; i < this->source_data_item_ptr_vec.size(); i++)
    {
        assert(this->source_data_item_ptr_vec[i] != NULL);
    }

    return this->source_data_item_ptr_vec;
}

vector<shared_ptr<data_item_record>> modify_vals_by_nnz_pad::get_dest_data_item_ptr_in_data_transform_step_without_check()
{
    // 执行过
    assert(this->is_run == true);
    // 内容符合要求
    assert(this->target_matrix_id >= 0);
    assert(this->nnz_target > 0);

    // 其中没有空指针
    for (unsigned long i = 0; i < this->dest_data_item_ptr_vec.size(); i++)
    {
        assert(this->dest_data_item_ptr_vec[i] != NULL);
    }

    return this->dest_data_item_ptr_vec;
}

string modify_vals_by_nnz_pad::convert_to_string()
{
    // 内容符合要求
    assert(this->target_matrix_id >= 0);
    string return_str = "modify_vals_by_nnz_pad::{name:\"" + name + "\",target_matrix_id:" + to_string(target_matrix_id) + "nnz_target:" + to_string(nnz_target) + "}";
    return return_str;
}