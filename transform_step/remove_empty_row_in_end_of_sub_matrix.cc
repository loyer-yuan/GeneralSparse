#include "../data_transform_step.hpp"

remove_empty_row_in_end_of_sub_matrix::remove_empty_row_in_end_of_sub_matrix(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id)
    : basic_data_transform_step("remove_empty_row_in_end_of_sub_matrix", meta_data_set_ptr)
{
    assert(meta_data_set_ptr != NULL);
    assert(meta_data_set_ptr->check());
    assert(target_matrix_id >= 0);

    this->target_matrix_id = target_matrix_id;
}

void remove_empty_row_in_end_of_sub_matrix::run(bool check)
{
    if (check)
    {
        assert(meta_data_set_ptr != NULL);
        assert(meta_data_set_ptr->check());
        assert(target_matrix_id >= 0);

        // 查看当前子块的行索引
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "nz_row_indices", this->target_matrix_id));
        // 查看子块的上下边界
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "begin_row_index", this->target_matrix_id));
        assert(this->meta_data_set_ptr->is_exist(GLOBAL_META, "end_row_index", this->target_matrix_id));
    }

    // 只要会改变下边界
    unsigned long begin_row_index = this->meta_data_set_ptr->get_element(GLOBAL_META, "begin_row_index", this->target_matrix_id)->get_metadata_arr()->read_integer_from_arr(0);
    unsigned long end_row_index = this->meta_data_set_ptr->get_element(GLOBAL_META, "end_row_index", this->target_matrix_id)->get_metadata_arr()->read_integer_from_arr(0);

    // 一共的行数量
    unsigned long row_num_of_sub_matrix = end_row_index - begin_row_index + 1;

    // 查看行索引
    shared_ptr<universal_array> row_index_ptr = this->meta_data_set_ptr->get_element(GLOBAL_META, "nz_row_indices", this->target_matrix_id)->get_metadata_arr();
    // 查看最后一个行索引的值，行索引必定是增序的，最后一行的行索引可以推断出尾部是不是有空行
    unsigned long max_row_index = row_index_ptr->read_integer_from_arr(row_index_ptr->get_len() - 1);

    if (check)
    {
        assert(max_row_index <= row_num_of_sub_matrix - 1);
    }

    if (max_row_index < row_num_of_sub_matrix - 1)
    {
        // 记录输入数据，行上边界、下边界和行索引。下边界是输出
        shared_ptr<data_item_record> begin_row_index_record(new data_item_record(GLOBAL_META, "begin_row_index", this->target_matrix_id));
        this->source_data_item_ptr_vec.push_back(begin_row_index_record);
        shared_ptr<data_item_record> end_row_index_record(new data_item_record(GLOBAL_META, "end_row_index", this->target_matrix_id));
        this->source_data_item_ptr_vec.push_back(end_row_index_record);
        shared_ptr<data_item_record> nz_row_indices_record(new data_item_record(GLOBAL_META, "nz_row_indices", this->target_matrix_id));
        this->source_data_item_ptr_vec.push_back(nz_row_indices_record);

        // 计算出新的下边界
        unsigned long new_end_row_index = begin_row_index + max_row_index;

        // 删除上边界，并且添加新的上边界
        this->meta_data_set_ptr->remove_element(GLOBAL_META, "end_row_index", this->target_matrix_id);
        // 添加新的下边界，增加一个新的end_row_index
        shared_ptr<meta_data_item> end_row_index_item_ptr(new meta_data_item(&new_end_row_index, UNSIGNED_LONG, GLOBAL_META, "end_row_index", this->target_matrix_id));
        this->meta_data_set_ptr->add_element(end_row_index_item_ptr);

        // 增加新的输出记录
        shared_ptr<data_item_record> end_row_index_record2(new data_item_record(GLOBAL_META, "end_row_index", this->target_matrix_id));
        this->dest_data_item_ptr_vec.push_back(end_row_index_record2);
    }

    this->is_run = true;
}

vector<shared_ptr<data_item_record>> remove_empty_row_in_end_of_sub_matrix::get_source_data_item_ptr_in_data_transform_step()
{
    assert(this->is_run == true);
    assert(this->target_matrix_id >= 0);

    for (unsigned long i = 0; i < this->source_data_item_ptr_vec.size(); i++)
    {
        // 不能空指针
        assert(this->source_data_item_ptr_vec[i] != NULL);
    }

    // 返回
    return this->source_data_item_ptr_vec;
}

vector<shared_ptr<data_item_record>> remove_empty_row_in_end_of_sub_matrix::get_dest_data_item_ptr_in_data_transform_step_without_check()
{
    assert(this->is_run == true);
    assert(this->target_matrix_id >= 0);

    for (unsigned long i = 0; i < this->dest_data_item_ptr_vec.size(); i++)
    {
        // 不能空指针
        assert(this->dest_data_item_ptr_vec[i] != NULL);
    }

    return this->dest_data_item_ptr_vec;
}

string remove_empty_row_in_end_of_sub_matrix::convert_to_string()
{
    assert(this->target_matrix_id >= 0);

    string return_str = "remove_empty_row_in_end_of_sub_matrix::{name:\"" + name + "\",target_matrix_id:" + to_string(target_matrix_id) + "}";

    return return_str;
}