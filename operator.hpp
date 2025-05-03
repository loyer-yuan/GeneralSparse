#ifndef OPERATOR_HPP
#define OPERATOR_HPP

#include <vector>
#include <memory>
#include <iostream>
#include <string>
#include "metadata_set.hpp"
#include "data_transform_step.hpp"
#include "code_generator.hpp"
#include "reduction_token.hpp"

using namespace std;

// 枚举operator的三个阶段
enum operator_stage_type
{
    CHOOSING_OP = -40,
    CONVERTING_OP ,
    DISTRIBUTING_OP,
    IMPLEMENTING_OP,
    NONE_OP,
};

// 检查operator stage的类型
bool check_operator_stage_type(operator_stage_type type);

// 打印具体的operator stage类型
string convert_operator_stage_type_to_string(operator_stage_type type);

// 用一个数据类型来存储operator的执行记录，包含每个data transform step的输入输出以及参数实例等
// 可以有多个来源，数据可以有多个源数据，但是只能有一个目标数据
class transform_step_record_item
{
public:
    transform_step_record_item(vector<shared_ptr<data_item_record>> source_data_item_ptr_vec, vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec, shared_ptr<basic_data_transform_step> transform_step_ptr);

    // 转化为字符串
    string convert_to_string();

    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_vec()
    {
        return source_data_item_ptr_vec;
    };

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_vec()
    {
        return dest_data_item_ptr_vec;
    };    
private:
    // 输入数据数据项
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    // 输出只能是一个
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;
    shared_ptr<basic_data_transform_step> transform_step_ptr;
};

// 根据一个transform step来获得一个对应的transform_step_record_item
shared_ptr<transform_step_record_item> get_record_item_of_a_transform_step(shared_ptr<basic_data_transform_step> transform_step_ptr);

class operator_context;

// 一些operator的
class basic_operator
{
public:
    basic_operator(string name, shared_ptr<meta_data_set> meta_data_set_ptr, operator_stage_type stage, int target_matrix_id)
    {
        assert(meta_data_set_ptr != NULL);
        this->name = name;
        this->meta_data_set_ptr = meta_data_set_ptr;
        this->stage = stage;
        this->target_matrix_id = target_matrix_id;
    }

    virtual void run(bool check = get_config()["OPERATOR_RUNTIME_CHECK"].as_bool())

    {
        cout << "basic_operator::run(): no actual operator" << endl;
        assert(false);
    }

    // 必须使用白名单的方式来规定一个operator是不是可以被使用
    // 检查是不是需要执行。因为执行过程是积极的，每次执行都是根据当前矩阵的实时状态来的
    virtual bool is_valid_according_to_metadata()
    {
        cout << "basic_operator::is_valid_according_to_metadata(): no actual operator" << endl;
        assert(false);
    }

    virtual bool is_valid_according_to_operator(shared_ptr<operator_context> operator_history = NULL)
    {
        cout << "basic_operator::is_valid_according_to_operator(): no actual operator" << endl;
        assert(false);
    }

    // 获取当前operator的执行记录，是一个数组，按照其执行的顺序排列到一个vector中
    virtual vector<shared_ptr<transform_step_record_item>> get_data_transform_sequence()
    {
        cout << "basic_operator::get_data_transform_sequence(): no actual operator" << endl;
        assert(false);
    }

    virtual bool check()
    {
        if (name == "default_operator_name" || stage == NONE_OP || this->meta_data_set_ptr != NULL)
        {
            return false;
        }

        if (check_operator_stage_type(stage) == false)
        {
            return false;
        }

        if (this->meta_data_set_ptr->check() == false)
        {
            return false;
        }

        return true;
    }
    //为tblock、warp行切分重执行使用
    virtual void set_padding_to_false()
    {
        cout << "basic_operator::set_padding_to_false(): no actual operator" << endl;
        assert(false);
    }

    virtual string convert_to_string()
    {
        cout << "basic_operator::convert_to_string(): no actual operator" << endl;
        assert(false);
    }

    virtual bool get_is_padding_with_col_size_in_bmt()
    {
        cout << "basic_operator::This is a specific pointer to BMT padding pattern: no actual GET function" << endl;
        assert(false);
    }

    virtual bool get_is_col_padding_with_row_max_size_with_empty_row()
    {
        cout << "basic_operator::This is a specific pointer to BMT padding pattern: no actual GET function" << endl;
        assert(false);
    }

    operator_stage_type get_stage()
    {
        return stage;
    }

    string get_name()
    {
        return name;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    void set_transform_seq(shared_ptr<transform_step_record_item> item)
    {
        vector<shared_ptr<data_item_record>> source = item->get_source_data_item_ptr_vec();
        vector<shared_ptr<data_item_record>> dest = item->get_dest_data_item_ptr_vec();
        if (source.size() == 0 && dest.size() == 0)
        {
            return;
        }
        else
        {
            this->transform_seq.push_back(item);
        }
    }
protected:
    // operator的名字
    string name = "default_operator_name";

    // operator的阶段
    operator_stage_type stage = NONE_OP;

    // operator要处理的对象
    shared_ptr<meta_data_set> meta_data_set_ptr;
    vector<shared_ptr<transform_step_record_item>> transform_seq;


    int target_matrix_id = -1;
};

class operator_context
{
public:
    operator_context();
    //读取执行器上下文中的某个数组
    vector<shared_ptr<basic_operator>> read_operator_context_arr(operator_stage_type type, int sub_matrix_id)
    {
        if (type == CONVERTING_OP)
        {
            return converting_operator;
        }
        else if (type == DISTRIBUTING_OP)
        {
            return distirbuting_operator[sub_matrix_id];
        }
        else if (type == IMPLEMENTING_OP)
        {
            return implementing_operator[sub_matrix_id];
        }
        else
        {
            cout << "operator::operator context: operator stage error" << endl;
            assert(false);
        }
    }

    //增加上下文
    void operator_context_add(shared_ptr<basic_operator> current_operator)
    {
        if (current_operator->get_stage() == CONVERTING_OP)
        {
            converting_operator.push_back(current_operator);
        }
        else if (current_operator->get_stage() == DISTRIBUTING_OP)
        {
            distirbuting_operator[current_operator->get_target_matrix_id()].push_back(current_operator);
        }
        else if (current_operator->get_stage() == IMPLEMENTING_OP)
        {
            implementing_operator[current_operator->get_target_matrix_id()].push_back(current_operator);
        }
        else
        {
            cout << "operator::operator_context:operator stage error" << endl;
            assert(false);
        }
    }

    //删除上下文
    void pop_back_context(operator_stage_type type, int sub_matrix_id)
    {
        if (type == CONVERTING_OP)
        {
            converting_operator.pop_back();
        }
        else if (type == DISTRIBUTING_OP)
        {
            distirbuting_operator[sub_matrix_id].pop_back();
        }
        else if (type == IMPLEMENTING_OP)
        {
            implementing_operator[sub_matrix_id].pop_back();
        }
        else
        {
            cout << "operator::operator context: operator stage error" << endl;
            assert(false);
        }
    }

private:
    vector<shared_ptr<basic_operator>> converting_operator;
    map<int, vector<shared_ptr<basic_operator>>> distirbuting_operator;
    map<int, vector<shared_ptr<basic_operator>>> implementing_operator;
};

// 执行全局排序表的operator
class sort_operator : public basic_operator
{
public:
    sort_operator(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id);
    sort_operator(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, shared_ptr<operator_context> operator_history);
    sort_operator(shared_ptr<code_generator> code_generator_ptr, shared_ptr<operator_context> operator_history);

    bool is_valid_according_to_metadata();
    bool is_valid_according_to_operator(shared_ptr<operator_context> operator_history = NULL);

    void run(bool check = get_config()["OPERATOR_RUNTIME_CHECK"].as_bool());

    vector<shared_ptr<transform_step_record_item>> get_data_transform_sequence();

    string convert_to_string();

private:
    bool is_run = false;
};

// 执行行切分，并且是固定间隔的切分
class fixed_interval_row_matrix_div_operator : public basic_operator
{
public:
    fixed_interval_row_matrix_div_operator(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int fixed_row_interval_size);
    fixed_interval_row_matrix_div_operator(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int fixed_row_interval_size, shared_ptr<operator_context> operator_history);
    fixed_interval_row_matrix_div_operator(shared_ptr<code_generator> code_generator_ptr, int fixed_row_interval_size, shared_ptr<operator_context> operator_history);

    bool is_valid_according_to_metadata();
    bool is_valid_according_to_operator(shared_ptr<operator_context> operator_history = NULL);

    void run(bool check = get_config()["OPERATOR_RUNTIME_CHECK"].as_bool());

    // 查看当前切分的间隔
    int get_fixed_row_interval_size()
    {
        return fixed_row_interval_size;
    }

    vector<shared_ptr<transform_step_record_item>> get_data_transform_sequence();

    string convert_to_string();

private:
    int fixed_row_interval_size = 0;

    // 查看有没有执行过
    bool is_run = false;
};

// 执行行切分，并且是固定间隔的切分
class row_nz_matrix_div_operator : public basic_operator
{
public:
    row_nz_matrix_div_operator(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int init_row_size_upper_boundary, int max_row_size_upper_boundary, int expansion_rate);
    row_nz_matrix_div_operator(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int init_row_size_upper_boundary, int max_row_size_upper_boundary, int expansion_rate, shared_ptr<operator_context> operator_history);
    row_nz_matrix_div_operator(shared_ptr<code_generator> code_generator_ptr, int init_row_size_upper_boundary, int max_row_size_upper_boundary, int expansion_rate, shared_ptr<operator_context> operator_history);

    bool is_valid_according_to_metadata();
    bool is_valid_according_to_operator(shared_ptr<operator_context> operator_history = NULL);

    void run(bool check = get_config()["OPERATOR_RUNTIME_CHECK"].as_bool());

    // 查看当前切分的间隔
    int get_init_row_size_upper_boundary()
    {
        return init_row_size_upper_boundary;
    }

    int get_expansion_rate()
    {
        return expansion_rate;
    }

    vector<shared_ptr<transform_step_record_item>> get_data_transform_sequence();

    string convert_to_string();

private:
    int init_row_size_upper_boundary = 0;
    int max_row_size_upper_boundary = 0;
    int expansion_rate = 0;

    // 查看有没有执行过
    bool is_run = false;
};

// 首先执行线程块粒度的行切分
class fixed_interval_row_direction_tblock_blocking_operator : public basic_operator
{
public:
    //is_padding: 行方向padding flag 
    fixed_interval_row_direction_tblock_blocking_operator(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int fixed_row_block_size, bool is_padding);
    fixed_interval_row_direction_tblock_blocking_operator(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int fixed_row_block_size, bool is_padding, shared_ptr<operator_context> operator_history);
    fixed_interval_row_direction_tblock_blocking_operator(shared_ptr<code_generator> code_generator_ptr, int fixed_row_block_size, bool is_padding, shared_ptr<operator_context> operator_history);

    bool is_valid_according_to_metadata();
    bool is_valid_according_to_operator(shared_ptr<operator_context> operator_history = NULL);

    void run(bool check = get_config()["OPERATOR_RUNTIME_CHECK"].as_bool());

    // 查看当前切分的间隔
    int get_fixed_row_block_size()
    {
        return fixed_row_block_size;
    }

    bool get_is_padding()
    {
        return is_padding;
    }

    void set_padding_to_false()
    {
        is_padding = false;
    }

    vector<shared_ptr<transform_step_record_item>> get_data_transform_sequence();

    string convert_to_string();

private:
    int fixed_row_block_size = 0;
    bool is_padding = false;
    shared_ptr<code_generator> code_generator_ptr = NULL;

    // 查看有没有执行过
    bool is_run = false;
};

class fixed_interval_row_direction_warp_blocking_operator : public basic_operator
{
public:
    //is_padding: 行方向padding flag 不能与relative 同时出现
    fixed_interval_row_direction_warp_blocking_operator(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int fixed_row_block_size, bool row_index_is_relative_to_BMTB, bool nz_index_is_relative_to_BMTB, bool is_padding);
    fixed_interval_row_direction_warp_blocking_operator(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int fixed_row_block_size, bool row_index_is_relative_to_BMTB, bool nz_index_is_relative_to_BMTB, bool is_padding, shared_ptr<operator_context> operator_history);
    fixed_interval_row_direction_warp_blocking_operator(shared_ptr<code_generator> code_generator_ptr, int fixed_row_block_size, bool row_index_is_relative_to_BMTB, bool nz_index_is_relative_to_BMTB, bool is_padding, shared_ptr<operator_context> operator_history);

    bool is_valid_according_to_metadata();
    bool is_valid_according_to_operator(shared_ptr<operator_context> operator_history = NULL);

    void run(bool check = get_config()["OPERATOR_RUNTIME_CHECK"].as_bool());

    // 查看当前切分的间隔
    int get_fixed_row_block_size()
    {
        return fixed_row_block_size;
    }

    bool get_nz_index_is_relative_to_BMTB()
    {
        return nz_index_is_relative_to_BMTB;
    }

    bool get_row_index_is_relative_to_BMTB()
    {
        return row_index_is_relative_to_BMTB;
    }

    void set_padding_to_false()
    {
        is_padding = false;
    }

    vector<shared_ptr<transform_step_record_item>> get_data_transform_sequence();

    string convert_to_string();

private:
    int fixed_row_block_size = 0;
    bool nz_index_is_relative_to_BMTB = false;
    bool row_index_is_relative_to_BMTB = false;
    // 查看是不是需要行方向的padding，当没有父块的时候才能执行
    bool is_padding = false;
    shared_ptr<code_generator> code_generator_ptr = NULL;

    // 查看有没有执行过
    bool is_run = false;
};

//执行空行padding的operator
class empty_row_pad_operator : public basic_operator
{
public:
    empty_row_pad_operator(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id);
    empty_row_pad_operator(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, shared_ptr<operator_context> operator_history);
    empty_row_pad_operator(shared_ptr<code_generator> code_generator_ptr, shared_ptr<operator_context> operator_history);

    bool is_valid_according_to_metadata();
    bool is_valid_according_to_operator(shared_ptr<operator_context> operator_history = NULL);

    void run(bool check = get_config()["OPERATOR_RUNTIME_CHECK"].as_bool());

    vector<shared_ptr<transform_step_record_item>> get_data_transform_sequence();

    string convert_to_string();

private:
    bool is_run = false;
};

// thread的列切分，如果没有父块，可以使用的是modify_row_indices(col_indices/vals)_by_col_pad_in_sub_matrix以及列切块
// target_matrix_id： 要执行列切分的子矩阵号
// fixed_col_block_size：列方向切分的宽度
// row(nz)_index_is_relative_to_BMTB(W)：查看是不是要生成相对索引
// is_padding_with_col_size_in_bmt：将每一行padding为fixed_col_block_size的整数倍，从而使得每一个BMT内部的非零元数量相同
// is_col_padding_with_row_max_size_without_empty_row：将父块中的每一行padding到一致，但是不包含空行
// padding_pos代表了具体要根据哪一个父块的max row来padding
// 因为padding会导致之前的内容全是错的，所以需要重执行之前的operator。
class fixed_interval_col_direction_thread_blocking_operator : public basic_operator
{
public:
    fixed_interval_col_direction_thread_blocking_operator(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id,
                                                          int fixed_col_block_size, bool row_index_is_relative_to_BMTB, bool nz_index_is_relative_to_BMTB, bool row_index_is_relative_to_BMW,
                                                          bool nz_index_is_relative_to_BMW, bool is_padding_with_col_size_in_bmt, bool is_col_padding_with_row_max_size_without_empty_row, POS_TYPE padding_pos, vector<shared_ptr<basic_operator>> former_operator);

    fixed_interval_col_direction_thread_blocking_operator(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id,
                                                          int fixed_col_block_size, bool row_index_is_relative,
                                                          bool nz_index_is_relative, bool is_padding_with_col_size_in_bmt, bool is_col_padding_with_row_max_size_without_empty_row, shared_ptr<operator_context> operator_history);
    
    fixed_interval_col_direction_thread_blocking_operator(shared_ptr<code_generator> code_generator_ptr,
                                                          int fixed_col_block_size, bool row_index_is_relative,
                                                          bool nz_index_is_relative, bool is_padding_with_col_size_in_bmt, bool is_col_padding_with_row_max_size_without_empty_row, shared_ptr<operator_context> operator_history);
    bool is_valid_according_to_metadata();
    bool is_valid_according_to_operator(shared_ptr<operator_context> operator_history = NULL);

    void run(bool check = get_config()["OPERATOR_RUNTIME_CHECK"].as_bool());

    // 查看当前切分的间隔
    int get_fixed_col_block_size()
    {
        return fixed_col_block_size;
    }

    bool get_nz_index_is_relative_to_BMTB()
    {
        return nz_index_is_relative_to_BMTB;
    }

    bool get_row_index_is_relative_to_BMTB()
    {
        return row_index_is_relative_to_BMTB;
    }

    bool get_row_index_is_relative_to_BMW()
    {
        return row_index_is_relative_to_BMW;
    }

    bool get_nz_index_is_relative_to_BMW()
    {
        return nz_index_is_relative_to_BMW;
    }

    bool get_is_padding_with_col_size_in_bmt()
    {
        return is_padding_with_col_size_in_bmt;
    }

    bool get_is_col_padding_with_row_max_size_without_empty_row()
    {
        return is_col_padding_with_row_max_size_without_empty_row;
    }

    POS_TYPE get_padding_pos()
    {
        return padding_pos;
    }

    vector<shared_ptr<basic_operator>> get_former_operator()
    {
        return former_operator;
    }

    vector<shared_ptr<transform_step_record_item>> get_data_transform_sequence();

    string convert_to_string();

private:
    int fixed_col_block_size = 0;

    // 相对索引
    bool nz_index_is_relative_to_BMTB = false;
    bool row_index_is_relative_to_BMTB = false;
    bool row_index_is_relative_to_BMW = false;
    bool nz_index_is_relative_to_BMW = false;

    // padding方式
    bool is_padding_with_col_size_in_bmt = false;
    bool is_col_padding_with_row_max_size_without_empty_row = false;
    POS_TYPE padding_pos = NONE_META;
    // padding导致了重执行 
    vector<shared_ptr<basic_operator>> former_operator;
    shared_ptr<code_generator> code_generator_ptr = NULL;

    // 查看有没有执行过
    bool is_run = false;
};

class interlance_storage_operator : public basic_operator
{
public:
    interlance_storage_operator(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, POS_TYPE pos);
    interlance_storage_operator(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, shared_ptr<operator_context> operator_history);
    interlance_storage_operator(shared_ptr<code_generator> code_generator_ptr, shared_ptr<operator_context> operator_history);

    bool is_valid_according_to_metadata();
    bool is_valid_according_to_operator(shared_ptr<operator_context> operator_history = NULL);

    void run(bool check = get_config()["OPERATOR_RUNTIME_CHECK"].as_bool());

    POS_TYPE get_pos()
    {
        return pos;
    }

    vector<shared_ptr<transform_step_record_item>> get_data_transform_sequence();

    string convert_to_string();

private:
    POS_TYPE pos = NONE_META;
    // 查看有没有执行过
    bool is_run = false;
};


// is_row_padding 判断 行方向 padding 不与 relative 混用
// is_col_padding_with_row_max_size_with_empty_row 列方向 padding 但是包含空行padding， 行切分后，保证生成的偏移为等差数列，为后续可能的压缩准备
// 同样需要重执行前面的切分
class fixed_interval_row_direction_thread_blocking_operator : public basic_operator
{
public:
    fixed_interval_row_direction_thread_blocking_operator(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int fixed_row_block_size, bool row_index_is_relative_to_parent, bool nz_index_is_relative_to_parent, bool is_row_padding, bool is_col_padding_with_row_max_size_with_empty_row, bool is_col_padding_with_col_size, int col_size, vector<shared_ptr<basic_operator>> former_operator);
    fixed_interval_row_direction_thread_blocking_operator(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int fixed_row_block_size, bool row_index_is_relative, bool nz_index_is_relative, bool is_row_padding, bool is_col_padding_with_row_max_size_with_empty_row, bool is_col_padding_with_col_size, int col_size, shared_ptr<operator_context> operator_history);
    fixed_interval_row_direction_thread_blocking_operator(shared_ptr<code_generator> code_generator_ptr, int fixed_row_block_size, bool row_index_is_relative, bool nz_index_is_relative, bool is_row_padding, bool is_col_padding_with_row_max_size_with_empty_row, bool is_col_padding_with_col_size, int col_size, shared_ptr<operator_context> operator_history);

    bool is_valid_according_to_metadata();
    bool is_valid_according_to_operator(shared_ptr<operator_context> operator_history = NULL);

    void run(bool check = get_config()["OPERATOR_RUNTIME_CHECK"].as_bool());

    // 查看当前切分的间隔
    int get_fixed_row_block_size()
    {
        return this->fixed_row_block_size;
    }

    bool get_nz_index_is_relative_to_parent()
    {
        return this->nz_index_is_relative_to_parent;
    }

    bool get_row_index_is_relative_to_parent()
    {
        return this->row_index_is_relative_to_parent;
    }

    bool get_is_col_padding_with_row_max_size_with_empty_row()
    {
        return this->is_col_padding_with_row_max_size_with_empty_row;
    }

    vector<shared_ptr<transform_step_record_item>> get_data_transform_sequence();

    string convert_to_string();

private:
    int fixed_row_block_size = 0;
    bool nz_index_is_relative_to_parent = false;
    bool row_index_is_relative_to_parent = false;
    // 查看是不是需要行方向的padding，当没有父块的时候才能执行
    bool is_row_padding = false;
    bool is_col_padding_with_row_max_size_with_empty_row = false;
    bool is_col_padding_with_col_size = false;
    int col_size = 1;
    // 查看有没有执行过
    vector<shared_ptr<basic_operator>> former_operator;
    shared_ptr<code_generator> code_generator_ptr = NULL;

    bool is_run = false;
};

//见thread列切分
class fixed_interval_col_direction_warp_blocking_operator : public basic_operator
{
public:
    fixed_interval_col_direction_warp_blocking_operator(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id,
                                                        int fixed_col_block_size, bool row_index_is_relative_to_BMTB, bool nz_index_is_relative_to_BMTB, bool is_padding_with_col_size_in_bmw, bool is_col_padding_with_row_max_size_without_empty_row, POS_TYPE padding_pos, vector<shared_ptr<basic_operator>> former_operator);

    fixed_interval_col_direction_warp_blocking_operator(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id,
                                                        int fixed_col_block_size, bool row_index_is_relative_to_BMTB, bool nz_index_is_relative_to_BMTB, bool is_padding_with_col_size_in_bmw, bool is_col_padding_with_row_max_size_without_empty_row, shared_ptr<operator_context> operator_history);

    fixed_interval_col_direction_warp_blocking_operator(shared_ptr<code_generator> code_generator_ptr,
                                                        int fixed_col_block_size, bool row_index_is_relative_to_BMTB, bool nz_index_is_relative_to_BMTB, bool is_padding_with_col_size_in_bmw, bool is_col_padding_with_row_max_size_without_empty_row, shared_ptr<operator_context> operator_history);
    bool is_valid_according_to_metadata();
    bool is_valid_according_to_operator(shared_ptr<operator_context> operator_history = NULL);

    void run(bool check = get_config()["OPERATOR_RUNTIME_CHECK"].as_bool());

    // 查看当前切分的间隔
    int get_fixed_col_block_size()
    {
        return fixed_col_block_size;
    }

    bool get_nz_index_is_relative_to_BMTB()
    {
        return nz_index_is_relative_to_BMTB;
    }

    bool get_row_index_is_relative_to_BMTB()
    {
        return row_index_is_relative_to_BMTB;
    }

    bool get_is_padding_with_col_size_in_bmw()
    {
        return is_padding_with_col_size_in_bmw;
    }

    bool get_is_col_padding_with_row_max_size_without_empty_row()
    {
        return is_col_padding_with_row_max_size_without_empty_row;
    }

    POS_TYPE get_padding_pos()
    {
        return padding_pos;
    }

    vector<shared_ptr<basic_operator>> get_former_operator()
    {
        return former_operator;
    }

    vector<shared_ptr<transform_step_record_item>> get_data_transform_sequence();

    string convert_to_string();

private:
    int fixed_col_block_size = 0;

    // 相对索引
    bool nz_index_is_relative_to_BMTB = false;
    bool row_index_is_relative_to_BMTB = false;

    // padding方式
    bool is_padding_with_col_size_in_bmw = false;
    bool is_col_padding_with_row_max_size_without_empty_row = false;
    POS_TYPE padding_pos = NONE_META;
    // padding导致了重执行
    vector<shared_ptr<basic_operator>> former_operator;
    shared_ptr<code_generator> code_generator_ptr = NULL;

    // 查看有没有执行过
    bool is_run = false;
};

class fixed_interval_col_direction_tblock_blocking_operator : public basic_operator
{
public:
    fixed_interval_col_direction_tblock_blocking_operator(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id,
                                                          int fixed_col_block_size, bool is_padding_with_col_size_in_bmtb, bool is_col_padding_with_row_max_size_without_empty_row);
    fixed_interval_col_direction_tblock_blocking_operator(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id,
                                                          int fixed_col_block_size, bool is_padding_with_col_size_in_bmtb, bool is_col_padding_with_row_max_size_without_empty_row, shared_ptr<operator_context> operator_history);
    fixed_interval_col_direction_tblock_blocking_operator(shared_ptr<code_generator> code_generator_ptr,
                                                          int fixed_col_block_size, bool is_padding_with_col_size_in_bmtb, bool is_col_padding_with_row_max_size_without_empty_row, shared_ptr<operator_context> operator_history);
    bool is_valid_according_to_metadata();
    bool is_valid_according_to_operator(shared_ptr<operator_context> operator_history = NULL);

    void run(bool check = get_config()["OPERATOR_RUNTIME_CHECK"].as_bool());

    // 查看当前切分的间隔
    int get_fixed_col_block_size()
    {
        return fixed_col_block_size;
    }

    bool get_is_padding_with_col_size_in_bmtb()
    {
        return is_padding_with_col_size_in_bmtb;
    }

    bool get_is_col_padding_with_row_max_size_without_empty_row()
    {
        return is_col_padding_with_row_max_size_without_empty_row;
    }

    vector<shared_ptr<transform_step_record_item>> get_data_transform_sequence();

    string convert_to_string();

private:
    int fixed_col_block_size = 0;

    // padding方式
    bool is_padding_with_col_size_in_bmtb = false;
    bool is_col_padding_with_row_max_size_without_empty_row = false;
    shared_ptr<code_generator> code_generator_ptr = NULL;

    // 查看有没有执行过
    bool is_run = false;

};




class balanced_interval_row_direction_thread_blocking_operator : public basic_operator
{
public:
    balanced_interval_row_direction_thread_blocking_operator(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int nnz_per_interval, bool row_index_is_relative_to_parent, bool nz_index_is_relative_to_parent);
    balanced_interval_row_direction_thread_blocking_operator(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int nnz_per_interval, bool row_index_is_relative, bool nz_index_is_relative, shared_ptr<operator_context> operator_history);
    balanced_interval_row_direction_thread_blocking_operator(shared_ptr<code_generator> code_generator_ptr, int nnz_per_interval, bool row_index_is_relative, bool nz_index_is_relative, shared_ptr<operator_context> operator_history);

    bool is_valid_according_to_metadata();
    bool is_valid_according_to_operator(shared_ptr<operator_context> operator_history = NULL);

    void run(bool check = get_config()["OPERATOR_RUNTIME_CHECK"].as_bool());

    // 查看当前切分的间隔
    int get_nnz_per_interval()
    {
        return nnz_per_interval;
    }

    bool get_nz_index_is_relative_to_parent()
    {
        return nz_index_is_relative_to_parent;
    }

    bool get_row_index_is_relative_to_parent()
    {
        return row_index_is_relative_to_parent;
    }


    vector<shared_ptr<transform_step_record_item>> get_data_transform_sequence();

    string convert_to_string();

private:
    int nnz_per_interval = 0;
    bool nz_index_is_relative_to_parent = false;
    bool row_index_is_relative_to_parent = false;

    shared_ptr<code_generator> code_generator_ptr = NULL;

    bool is_run = false;
};




class balanced_interval_row_direction_warp_blocking_operator : public basic_operator
{
public:
    balanced_interval_row_direction_warp_blocking_operator(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int nnz_per_interval, bool row_index_is_relative_to_BMTB, bool nz_index_is_relative_to_BMTB);
    balanced_interval_row_direction_warp_blocking_operator(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int nnz_per_interval, bool row_index_is_relative, bool nz_index_is_relative, shared_ptr<operator_context> operator_history);
    balanced_interval_row_direction_warp_blocking_operator(shared_ptr<code_generator> code_generator_ptr, int nnz_per_interval, bool row_index_is_relative, bool nz_index_is_relative, shared_ptr<operator_context> operator_history);

    bool is_valid_according_to_metadata();
    bool is_valid_according_to_operator(shared_ptr<operator_context> operator_history = NULL);

    void run(bool check = get_config()["OPERATOR_RUNTIME_CHECK"].as_bool());

    // 查看当前切分的间隔
    int get_nnz_per_interval()
    {
        return nnz_per_interval;
    }

    bool get_nz_index_is_relative_to_BMTB()
    {
        return nz_index_is_relative_to_BMTB;
    }

    bool get_row_index_is_relative_to_BMTB()
    {
        return row_index_is_relative_to_BMTB;
    }

    void set_padding_to_false()
    {
        
    }
    vector<shared_ptr<transform_step_record_item>> get_data_transform_sequence();

    string convert_to_string();

private:
    int nnz_per_interval = 0;
    bool nz_index_is_relative_to_BMTB = false;
    bool row_index_is_relative_to_BMTB = false;
    shared_ptr<code_generator> code_generator_ptr = NULL;

    bool is_run = false;
};




class balanced_interval_row_direction_tblock_blocking_operator : public basic_operator
{
public:
    balanced_interval_row_direction_tblock_blocking_operator(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int nnz_per_interval);
    balanced_interval_row_direction_tblock_blocking_operator(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int nnz_per_interval, shared_ptr<operator_context> operator_history);
    balanced_interval_row_direction_tblock_blocking_operator(shared_ptr<code_generator> code_generator_ptr, int nnz_per_interval,  shared_ptr<operator_context> operator_history);

    bool is_valid_according_to_metadata();
    bool is_valid_according_to_operator(shared_ptr<operator_context> operator_history = NULL);

    void run(bool check = get_config()["OPERATOR_RUNTIME_CHECK"].as_bool());

    // 查看当前切分的间隔
    int get_nnz_per_interval()
    {
        return nnz_per_interval;
    }

    void set_padding_to_false()
    {
        
    }
    
    vector<shared_ptr<transform_step_record_item>> get_data_transform_sequence();

    string convert_to_string();

private:
    int nnz_per_interval = 0;
    shared_ptr<code_generator> code_generator_ptr = NULL;
    bool is_run = false;

};


class fixed_interval_nnz_direction_thread_blocking_operator : public basic_operator
{
public:
    fixed_interval_nnz_direction_thread_blocking_operator(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int nnz_per_BMT, bool row_index_is_relative_to_parent, bool nz_index_is_relative_to_parent, bool nnz_padding);
    fixed_interval_nnz_direction_thread_blocking_operator(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int nnz_per_BMT, bool row_index_is_relative_to_parent, bool nz_index_is_relative_to_parent, bool nnz_padding, shared_ptr<operator_context> operator_history);
    fixed_interval_nnz_direction_thread_blocking_operator(shared_ptr<code_generator> code_generator_ptr, int nnz_per_BMT, bool row_index_is_relative_to_parent, bool nz_index_is_relative_to_parent, bool nnz_padding, shared_ptr<operator_context> operator_history);

    bool is_valid_according_to_metadata();
    bool is_valid_according_to_operator(shared_ptr<operator_context> operator_history = NULL);

    void run(bool check = get_config()["OPERATOR_RUNTIME_CHECK"].as_bool());

    // 查看当前切分的间隔
    int get_nnz_per_BMT()
    {
        return nnz_per_BMT;
    }

    vector<shared_ptr<transform_step_record_item>> get_data_transform_sequence();

    string convert_to_string();

private:
    int nnz_per_BMT = 0;
    shared_ptr<code_generator> code_generator_ptr = NULL;

    bool is_run = false;
    bool row_index_is_relative_to_parent = false;
    bool nz_index_is_relative_to_parent = false;
    bool nnz_padding = false;
};



class fixed_interval_nnz_direction_warp_blocking_operator : public basic_operator
{
public:
    fixed_interval_nnz_direction_warp_blocking_operator(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int nnz_per_BMW, bool row_index_is_relative_to_parent, bool nz_index_is_relative_to_parent, bool nnz_padding);
    fixed_interval_nnz_direction_warp_blocking_operator(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int nnz_per_BMW, bool row_index_is_relative_to_parent, bool nz_index_is_relative_to_parent, bool nnz_padding, shared_ptr<operator_context> operator_history);
    fixed_interval_nnz_direction_warp_blocking_operator(shared_ptr<code_generator> code_generator_ptr, int nnz_per_BMW, bool row_index_is_relative_to_parent, bool nz_index_is_relative_to_parent, bool nnz_padding, shared_ptr<operator_context> operator_history);

    bool is_valid_according_to_metadata();
    bool is_valid_according_to_operator(shared_ptr<operator_context> operator_history = NULL);

    void run(bool check = get_config()["OPERATOR_RUNTIME_CHECK"].as_bool());

    // 查看当前切分的间隔
    int get_nnz_per_BMW()
    {
        return nnz_per_BMW;
    }

    vector<shared_ptr<transform_step_record_item>> get_data_transform_sequence();

    string convert_to_string();

private:
    int nnz_per_BMW = 0;
    shared_ptr<code_generator> code_generator_ptr = NULL;

    bool is_run = false;
    bool row_index_is_relative_to_parent = false;
    bool nz_index_is_relative_to_parent = false;
    bool nnz_padding = false;
};


class fixed_interval_nnz_direction_tblock_blocking_operator : public basic_operator
{
public:
    fixed_interval_nnz_direction_tblock_blocking_operator(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int nnz_per_BMTB, bool nnz_padding);
    fixed_interval_nnz_direction_tblock_blocking_operator(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int nnz_per_BMTB, bool nnz_padding, shared_ptr<operator_context> operator_history);
    fixed_interval_nnz_direction_tblock_blocking_operator(shared_ptr<code_generator> code_generator_ptr, int nnz_per_BMTB, bool nnz_padding, shared_ptr<operator_context> operator_history);

    bool is_valid_according_to_metadata();
    bool is_valid_according_to_operator(shared_ptr<operator_context> operator_history = NULL);

    void run(bool check = get_config()["OPERATOR_RUNTIME_CHECK"].as_bool());

    // 查看当前切分的间隔
    int get_nnz_per_BMTB()
    {
        return nnz_per_BMTB;
    }

    vector<shared_ptr<transform_step_record_item>> get_data_transform_sequence();

    string convert_to_string();

private:
    int nnz_per_BMTB = 0;
    shared_ptr<code_generator> code_generator_ptr = NULL;

    bool is_run = false;
    bool nnz_padding = false;
};


class calculation_method_choose_operator : public basic_operator
{
public:
    calculation_method_choose_operator(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int method);
    calculation_method_choose_operator(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int method, shared_ptr<operator_context> operator_history);
    calculation_method_choose_operator(shared_ptr<code_generator> code_generator_ptr, int method, shared_ptr<operator_context> operator_history);

    bool is_valid_according_to_metadata();
    bool is_valid_according_to_operator(shared_ptr<operator_context> operator_history = NULL);

    void run(bool check = get_config()["OPERATOR_RUNTIME_CHECK"].as_bool());

    vector<shared_ptr<transform_step_record_item>> get_data_transform_sequence();

    string convert_to_string();

private:
    bool is_run = false;
    int method = -1;
};


class col_based_sort_operator : public basic_operator
{
public:
    col_based_sort_operator(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id);
    col_based_sort_operator(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, shared_ptr<operator_context> operator_history);
    col_based_sort_operator(shared_ptr<code_generator> code_generator_ptr, shared_ptr<operator_context> operator_history);

    bool is_valid_according_to_metadata();
    bool is_valid_according_to_operator(shared_ptr<operator_context> operator_history = NULL);

    void run(bool check = get_config()["OPERATOR_RUNTIME_CHECK"].as_bool());

    vector<shared_ptr<transform_step_record_item>> get_data_transform_sequence();

    string convert_to_string();

private:
    bool is_run = false;
};


class grid_block_operator : public basic_operator
{
public:

    grid_block_operator(shared_ptr<code_generator> code_generator_ptr, unsigned int grid_x, vector<unsigned int> block, unsigned int coarsen_factor, shared_ptr<operator_context> operator_history);

    bool is_valid_according_to_metadata();
    bool is_valid_according_to_operator(shared_ptr<operator_context> operator_history = NULL);

    void run(bool check = get_config()["OPERATOR_RUNTIME_CHECK"].as_bool());

    vector<shared_ptr<transform_step_record_item>> get_data_transform_sequence();

    string convert_to_string();

private:
    bool is_run = false;
    shared_ptr<code_generator> code_generator_ptr;
    vector<unsigned int> grid;
    vector<unsigned int> block;
};


class thread_total_reduce_operator : public basic_operator
{
public:

    thread_total_reduce_operator(shared_ptr<code_generator> code_generator_ptr, bool need_warp_reduction, unsigned int sparse_coarsen_factor, unsigned int coarsen_factor,  shared_ptr<operator_context> operator_history);

    bool is_valid_according_to_metadata();
    bool is_valid_according_to_operator(shared_ptr<operator_context> operator_history = NULL);

    void run(bool check = get_config()["OPERATOR_RUNTIME_CHECK"].as_bool());

    vector<shared_ptr<transform_step_record_item>> get_data_transform_sequence();

    string convert_to_string();

private:
    bool is_run = false;
    shared_ptr<code_generator> code_generator_ptr;
    bool need_warp_reduction;
    unsigned int coarsen_factor;
    unsigned int sparse_coarsen_factor;
};

class thread_bit_map_operator : public basic_operator
{
public:

    thread_bit_map_operator(shared_ptr<code_generator> code_generator_ptr, POS_TYPE pos, unsigned int size, unsigned int sparse_coarsen_factor, unsigned int coarsen_factor,  shared_ptr<operator_context> operator_history);

    bool is_valid_according_to_metadata();
    bool is_valid_according_to_operator(shared_ptr<operator_context> operator_history = NULL);

    void run(bool check = get_config()["OPERATOR_RUNTIME_CHECK"].as_bool());

    vector<shared_ptr<transform_step_record_item>> get_data_transform_sequence();

    string convert_to_string();

private:
    bool is_run = false;
    shared_ptr<code_generator> code_generator_ptr;
    POS_TYPE pos;
    unsigned int size;
    unsigned int coarsen_factor;
    unsigned int sparse_coarsen_factor;
};

class warp_total_reduce_operator : public basic_operator
{
public:

    warp_total_reduce_operator(shared_ptr<code_generator> code_generator_ptr, unsigned int coarsen_factor,  shared_ptr<operator_context> operator_history);

    bool is_valid_according_to_metadata();
    bool is_valid_according_to_operator(shared_ptr<operator_context> operator_history = NULL);

    void run(bool check = get_config()["OPERATOR_RUNTIME_CHECK"].as_bool());

    vector<shared_ptr<transform_step_record_item>> get_data_transform_sequence();

    string convert_to_string();

private:
    bool is_run = false;
    shared_ptr<code_generator> code_generator_ptr;
    unsigned int coarsen_factor;
};


class warp_bit_map_operator : public basic_operator
{
public:

    warp_bit_map_operator(shared_ptr<code_generator> code_generator_ptr, unsigned int coarsen_factor, bool relative_nz, bool relative_row, shared_ptr<operator_context> operator_history);

    bool is_valid_according_to_metadata();
    bool is_valid_according_to_operator(shared_ptr<operator_context> operator_history = NULL);

    void run(bool check = get_config()["OPERATOR_RUNTIME_CHECK"].as_bool());

    vector<shared_ptr<transform_step_record_item>> get_data_transform_sequence();

    string convert_to_string();

private:
    bool is_run = false;
    bool relative_nz = false;
    bool relative_row = false;
    shared_ptr<code_generator> code_generator_ptr;
    unsigned int coarsen_factor;
};

class warp_segment_reduce_operator : public basic_operator
{
public:

    warp_segment_reduce_operator(shared_ptr<code_generator> code_generator_ptr, unsigned int coarsen_factor, bool relative_nz, bool relative_row, shared_ptr<operator_context> operator_history);

    bool is_valid_according_to_metadata();
    bool is_valid_according_to_operator(shared_ptr<operator_context> operator_history = NULL);

    void run(bool check = get_config()["OPERATOR_RUNTIME_CHECK"].as_bool());

    vector<shared_ptr<transform_step_record_item>> get_data_transform_sequence();

    string convert_to_string();

private:
    bool is_run = false;
    bool relative_nz = false;
    bool relative_row = false;
    shared_ptr<code_generator> code_generator_ptr;
    unsigned int coarsen_factor;
};


class tblock_total_reduce_operator : public basic_operator
{
public:

    tblock_total_reduce_operator(shared_ptr<code_generator> code_generator_ptr, unsigned int coarsen_factor,  shared_ptr<operator_context> operator_history);

    bool is_valid_according_to_metadata();
    bool is_valid_according_to_operator(shared_ptr<operator_context> operator_history = NULL);

    void run(bool check = get_config()["OPERATOR_RUNTIME_CHECK"].as_bool());

    vector<shared_ptr<transform_step_record_item>> get_data_transform_sequence();

    string convert_to_string();

private:
    bool is_run = false;
    shared_ptr<code_generator> code_generator_ptr;
    unsigned int coarsen_factor;
};



class tblock_thread_bit_map_operator : public basic_operator
{
public:

    tblock_thread_bit_map_operator(shared_ptr<code_generator> code_generator_ptr, unsigned int coarsen_factor,  int block_size, bool relative_nz, bool relative_row, shared_ptr<operator_context> operator_history);

    bool is_valid_according_to_metadata();
    bool is_valid_according_to_operator(shared_ptr<operator_context> operator_history = NULL);

    void run(bool check = get_config()["OPERATOR_RUNTIME_CHECK"].as_bool());

    vector<shared_ptr<transform_step_record_item>> get_data_transform_sequence();

    string convert_to_string();

private:
    bool is_run = false;
    int block_size = 0;
    bool relative_nz = false;
    bool relative_row = false;
    shared_ptr<code_generator> code_generator_ptr;
    unsigned int coarsen_factor;
};




class merge_path_tblock_operator : public basic_operator
{
public:
    //is_padding: 行方向padding flag 
    merge_path_tblock_operator(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int work_size);
    merge_path_tblock_operator(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int work_size, shared_ptr<operator_context> operator_history);
    merge_path_tblock_operator(shared_ptr<code_generator> code_generator_ptr, int work_size, shared_ptr<operator_context> operator_history);

    bool is_valid_according_to_metadata();
    bool is_valid_according_to_operator(shared_ptr<operator_context> operator_history = NULL);

    void run(bool check = get_config()["OPERATOR_RUNTIME_CHECK"].as_bool());

    // 查看当前切分的间隔
    int get_work_size()
    {
        return work_size;
    }

    vector<shared_ptr<transform_step_record_item>> get_data_transform_sequence();

    string convert_to_string();

private:
    int work_size = 0;
    shared_ptr<code_generator> code_generator_ptr = NULL;
    bool is_run = false;
};



class merge_path_warp_operator : public basic_operator
{
public:
    //is_padding: 行方向padding flag 
    merge_path_warp_operator(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int work_size);
    merge_path_warp_operator(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int work_size, shared_ptr<operator_context> operator_history);
    merge_path_warp_operator(shared_ptr<code_generator> code_generator_ptr, int work_size, shared_ptr<operator_context> operator_history);

    bool is_valid_according_to_metadata();
    bool is_valid_according_to_operator(shared_ptr<operator_context> operator_history = NULL);

    void run(bool check = get_config()["OPERATOR_RUNTIME_CHECK"].as_bool());

    // 查看当前切分的间隔
    int get_work_size()
    {
        return work_size;
    }

    vector<shared_ptr<transform_step_record_item>> get_data_transform_sequence();

    string convert_to_string();

private:
    int work_size = 0;
    shared_ptr<code_generator> code_generator_ptr = NULL;
    bool is_run = false;
};


class merge_path_thread_operator : public basic_operator
{
public:
    //is_padding: 行方向padding flag 
    merge_path_thread_operator(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int work_size);
    merge_path_thread_operator(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int work_size, shared_ptr<operator_context> operator_history);
    merge_path_thread_operator(shared_ptr<code_generator> code_generator_ptr, int work_size, shared_ptr<operator_context> operator_history);

    bool is_valid_according_to_metadata();
    bool is_valid_according_to_operator(shared_ptr<operator_context> operator_history = NULL);

    void run(bool check = get_config()["OPERATOR_RUNTIME_CHECK"].as_bool());

    // 查看当前切分的间隔
    int get_work_size()
    {
        return work_size;
    }

    vector<shared_ptr<transform_step_record_item>> get_data_transform_sequence();

    string convert_to_string();

private:
    int work_size = 0;
    shared_ptr<code_generator> code_generator_ptr = NULL;
    bool is_run = false;
};







// class sparse_dense_div : public basic_operator
// {
// public:
//     sparse_dense_div(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int fixed_row_interval_size);
//     sparse_dense_div(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int fixed_row_interval_size, shared_ptr<operator_context> operator_history);

//     bool is_valid_according_to_metadata();
//     bool is_valid_according_to_operator(shared_ptr<operator_context> operator_history = NULL);

//     void run(bool check = get_config()["OPERATOR_RUNTIME_CHECK"].as_bool());

//     // 查看当前切分的间隔
//     int get_fixed_row_interval_size()
//     {
//         return fixed_row_interval_size;
//     }

//     vector<shared_ptr<transform_step_record_item>> get_data_transform_sequence();

//     string convert_to_string();

// private:
//     int fixed_row_interval_size = 0;

//     // 查看有没有执行过
//     bool is_run = false;
// };




#endif