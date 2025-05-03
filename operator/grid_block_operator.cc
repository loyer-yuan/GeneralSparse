#include "../operator.hpp"
grid_block_operator::grid_block_operator(shared_ptr<code_generator> code_generator_ptr,unsigned int grid_x, vector<unsigned int> block, unsigned int coarsen_factor, shared_ptr<operator_context> operator_history)
    : basic_operator("grid_block_operator", code_generator_ptr->get_metadata_set(), IMPLEMENTING_OP, code_generator_ptr->get_sub_matrix_id())
{
    // if (block[0] == 1)
    // {
    //     assert(block[1] % 32 == 0);
    // }
    // else
    // {
    //     assert(block[0] % 32 == 0);
    // }
    if (block[1] % 2 != 0)
    {
        block[1] = block[1] + 1;
    }

    unsigned int grid_y;
    if (this->meta_data_set_ptr->is_exist(WARP_META, "first_nz_indices", this->target_matrix_id) == false)
    {
        grid_y = (get_config()["DENSE_MATRIX_SIZE"].as_integer() % (block[0] * coarsen_factor) == 0) ? get_config()["DENSE_MATRIX_SIZE"].as_integer() / (block[0] * coarsen_factor) : get_config()["DENSE_MATRIX_SIZE"].as_integer() / (block[0] * coarsen_factor) + 1;
    }
    else
    {
        grid_y = (get_config()["DENSE_MATRIX_SIZE"].as_integer() % (block[1] * coarsen_factor) == 0) ? get_config()["DENSE_MATRIX_SIZE"].as_integer() / (block[1] * coarsen_factor) : get_config()["DENSE_MATRIX_SIZE"].as_integer() / (block[1] * coarsen_factor) + 1;
    }
    vector<unsigned int> grid_vec;
    grid_vec.push_back(grid_x);
    grid_vec.push_back(grid_y);

    this->grid = grid_vec;
    this->block = block;
    this->code_generator_ptr = code_generator_ptr;
}

bool grid_block_operator::is_valid_according_to_operator(shared_ptr<operator_context> operator_history)
{
    return true;
}

bool grid_block_operator::is_valid_according_to_metadata()
{
    return true;
}

// 执行具体的排序操作
void grid_block_operator::run(bool check)
{
    this->code_generator_ptr->set_thread_grid(this->grid, this->block);
}

// 给出当前op的操作序列，用以生成format conversion
vector<shared_ptr<transform_step_record_item>> grid_block_operator::get_data_transform_sequence()
{
    assert(this->target_matrix_id >= 0);

    for (unsigned long i = 0; i < this->transform_seq.size(); i++)
    {
        assert(this->transform_seq[i] != NULL);
    }

    // 返回所有的操作集合
    return this->transform_seq;
}

string grid_block_operator::convert_to_string()
{
    assert(this->target_matrix_id >= 0);

    string return_str = "grid_block_operator::{name:\"" + this->name + "\",stage:" + convert_operator_stage_type_to_string(this->stage) + ",target_matrix_id:" + to_string(target_matrix_id) + "}";

    return return_str;
}