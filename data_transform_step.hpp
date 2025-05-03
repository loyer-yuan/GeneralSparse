#ifndef DATA_TRANSFORM_STEP_HPP
#define DATA_TRANSFORM_STEP_HPP

#include <vector>
#include <memory>
#include <iostream>
#include <string>
#include "metadata_set.hpp"
#include "data_transform_common.hpp"
#include <algorithm>
#include <set>

using namespace std;

// 用来表达一个输入输出的数据项，近似于构成meta_data名字的三元组
// 用来界定data_transform_step的输入和输出
class data_item_record
{
public:
    data_item_record(POS_TYPE meta_position, string name, int sub_matrix_id);

    // 将一个数据记录转换为一个字符串
    string convert_to_string();

    // 数据所属于的子矩阵
    int get_sub_matrix_id()
    {
        return sub_matrix_id;
    }

private:
    POS_TYPE meta_position = NONE_META;
    // 当前的matadata的名字
    string name = "none";
    // 查看元数据的类型，数组还是常量
    // META_TYPE metadata_type = NONE_META_TYPE;
    // 查看当前数组所在的子矩阵，最终会变成变量名的后缀，-1代表是未分块的全局原始矩阵的基本信息
    int sub_matrix_id = -1;
};

// data_transform_step
class data_transform_step
{
public:
    data_transform_step(string name, shared_ptr<meta_data_set> meta_data_set_ptr)
    {
        assert(meta_data_set_ptr != NULL);
        assert(meta_data_set_ptr->check());
        this->name = name;
        this->meta_data_set_ptr = meta_data_set_ptr;
    }

    // 真正执行
    virtual void run(bool check = true)
    {
        cout << "data_transform_step::run(): no actual data_transform_step" << endl;
        assert(false);
    }

    // 获取当前数据转换步骤（data_transform_step）的输入数据条目，只有在run之后才能得到输入和输出的数据记录
    virtual vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step()
    {
        cout << "data_transform_step::get_source_data_item_ptr_in_data_transform_step(): no actual data_transform_step" << endl;
        assert(false);
        vector<shared_ptr<data_item_record>> return_vec;
        return return_vec;
    }

    // 获取当前data_transform_step的输出度条目，只有在run之后才能得到输入和输出的数据记录
    virtual vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step()
    {
        vector<shared_ptr<data_item_record>> return_vec = get_dest_data_item_ptr_in_data_transform_step_without_check();

        return return_vec;
    }

    virtual bool check()
    {
        if (this->name == "default_data_transform_step_name" || this->meta_data_set_ptr == NULL)
        {
            return false;
        }

        if (this->meta_data_set_ptr->check() == false)
        {
            return false;
        }

        return true;
    }

    // 将一个数据记录转换为一个字符串
    virtual string convert_to_string()
    {
        cout << "data_transform_step::convert_to_string(): no actual data_transform_step" << endl;
        assert(false);
        return NULL;
    }

    // bool get_is_run()
    // {
    //     return is_run;
    // }
protected:
    // 当前transform操作的名字，在命名中主要关注对于对应metada的真正操作。
    string name = "default_data_transform_step_name";
    // 当前要修改的metadata set
    shared_ptr<meta_data_set> meta_data_set_ptr;
    // 用一个变量看当前transform是不是已经被执行过，只有执行过才能给出输入输出的记录
    bool is_run = false;

    // 需要一个函数来被实现和获得输出数据的记录，这个是必须被实现的额
    virtual vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check()
    {
        cout << "data_transform_step::get_dest_data_item_ptr_in_data_transform_step_without_check(): no actual data_transform_step" << endl;
        assert(false);
    }
};

// merge_transform_step
class merge_data_transform_step : public data_transform_step
{
public:
    merge_data_transform_step(string name, shared_ptr<meta_data_set> meta_data_set_ptr) : data_transform_step(name, meta_data_set_ptr)
    {
    }

    // 真正执行
    virtual void run(bool check = true)
    {
        cout << "merge_data_transform_step::run(): no actual merge_data_transform_step" << endl;
        assert(false);
    }

    // 获取当前数据转换步骤（data_transform_step）的输入数据条目，只有在run之后才能得到输入和输出的数据记录
    virtual vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step()
    {
        cout << "merge_data_transform_step::get_source_data_item_ptr_in_data_transform_step(): no actual merge_data_transform_step" << endl;
        assert(false);
        vector<shared_ptr<data_item_record>> return_vec;
        return return_vec;
    }

    // 获取当前data_transform_step的输出度条目，只有在run之后才能得到输入和输出的数据记录
    virtual vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step()
    {
        vector<shared_ptr<data_item_record>> return_vec = get_dest_data_item_ptr_in_data_transform_step_without_check();

        return return_vec;
    }

    virtual bool check()
    {
        if (this->name == "default_data_transform_step_name" || this->meta_data_set_ptr == NULL)
        {
            return false;
        }

        if (this->meta_data_set_ptr->check() == false)
        {
            return false;
        }

        return true;
    }

    // 将一个数据记录转换为一个字符串
    virtual string convert_to_string()
    {
        cout << "merge_data_transform_step::convert_to_string(): no actual merge_data_transform_step" << endl;
        assert(false);
        return NULL;
    }
};

class basic_data_transform_step : public data_transform_step
{
public:
    basic_data_transform_step(string name, shared_ptr<meta_data_set> meta_data_set_ptr) : data_transform_step(name, meta_data_set_ptr)
    {
    }

    // 真正执行
    virtual void run(bool check = true)
    {
        cout << "basic_data_transform_step::run(): no actual basic_data_transform_step" << endl;
        assert(false);
    }

    // 获取当前数据转换步骤（data_transform_step）的输入数据条目，只有在run之后才能得到输入和输出的数据记录
    virtual vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step()
    {
        cout << "basic_data_transform_step::get_source_data_item_ptr_in_data_transform_step(): no actual basic_data_transform_step" << endl;
        assert(false);
        vector<shared_ptr<data_item_record>> return_vec;
        return return_vec;
    }

    // 获取当前data_transform_step的输出度条目，只有在run之后才能得到输入和输出的数据记录
    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step()
    {
        // assert(this->is_run == true); 具体问题具体分析，不需要
        // 这个是带有对输出条目进行检查的函数
        vector<shared_ptr<data_item_record>> return_vec = get_dest_data_item_ptr_in_data_transform_step_without_check();

        // 所有的输出数据记录，需要所有的输出来自于不一样的子块
        set<int> target_matrix_id_set;

        // 遍历，查看输出数据所属于的
        for (unsigned long i = 0; i < return_vec.size(); i++)
        {
            int target_matrix_id = return_vec[i]->get_sub_matrix_id();

            // 当前输出数据所属于的子矩阵一定不存在
            assert(target_matrix_id_set.count(target_matrix_id) == 0);

            target_matrix_id_set.insert(target_matrix_id);
        }

        return return_vec;
    }

    virtual bool check()
    {
        if (this->name == "default_data_transform_step_name" || this->meta_data_set_ptr == NULL)
        {
            return false;
        }

        if (this->meta_data_set_ptr->check() == false)
        {
            return false;
        }

        return true;
    }

    // 将一个数据记录转换为一个字符串
    virtual string convert_to_string()
    {
        cout << "basic_data_transform_step::convert_to_string(): no actual basic_data_transform_step" << endl;
        assert(false);
        return NULL;
    }
};

// 通过nz_row_indices，获得每一行的非零元数量，并且执行排序。这里也许有一个按照范围排序的做法
// 也就是现在行方向分块，然后针对每一个行条带内部执行排序的结果，这里先不考虑
class get_row_order_by_length : public basic_data_transform_step
{
public:
    // 构造函数
    get_row_order_by_length(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id);

    // 真正执行
    void run(bool check = true);

    int get_target_matrix_id()
    {
        return this->target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    // shared_ptr<data_item_record> get_dest_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    // 需要被排序的矩阵号，矩阵被切分会导致产生多个矩阵
    // 矩阵号是不能被二次修改的
    int target_matrix_id = -1;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class get_row_order_by_col : public basic_data_transform_step
{
public:
    // 构造函数
    get_row_order_by_col(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id);

    // 真正执行
    void run(bool check = true);

    int get_target_matrix_id()
    {
        return this->target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

// 对子矩阵的行索引进行重排，重排依赖于排序前与排序后的行索引映射，也在metadata set中
class reorder_row_by_index : public basic_data_transform_step
{
public:
    // 构造函数
    reorder_row_by_index(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id);

    // 真正执行
    void run(bool check = true);

    int get_target_matrix_id()
    {
        return this->target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    // shared_ptr<data_item_record> get_dest_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

// 根据未重排的行索引，对子矩阵的列索引进行重排，重排依赖于排序前与排序后的行索引映射，也在metadata set中
class reorder_col_by_index : public basic_data_transform_step
{
public:
    reorder_col_by_index(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id);

    // 真正执行
    void run(bool check = true);

    int get_target_matrix_id()
    {
        return this->target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    // shared_ptr<data_item_record> get_dest_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

// 根据未重排的行索引，对子矩阵的列索引进行重排，重排依赖于排序前与排序后的行索引映射，也在metadata set中
class reorder_val_by_index : public basic_data_transform_step
{
public:
    reorder_val_by_index(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id);

    // 真正执行
    void run(bool check = true);

    int get_target_matrix_id()
    {
        return this->target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    // shared_ptr<data_item_record> get_dest_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

// 根据非零元的行索引对列索引执行分块，被分块的子矩阵和分块的间隔可以作为参数
class fixed_div_col_indices_by_corr_row_indices : public basic_data_transform_step
{
public:
    fixed_div_col_indices_by_corr_row_indices(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int fixed_row_gap_size);

    // 真正执行
    void run(bool check = true);

    unsigned long get_fixed_row_gap_size()
    {
        return this->fixed_row_gap_size;
    }

    int get_target_matrix_id()
    {
        return this->target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    // shared_ptr<data_item_record> get_dest_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    unsigned long fixed_row_gap_size = 0;

    // 使用两个数组分别记录输入数据记录和输出数据记录
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

// 根据非零元的行索引对值数组执行分块，被分块的子矩阵和分块的间隔可以作为参数
class fixed_div_vals_by_corr_row_indices : public basic_data_transform_step
{
public:
    fixed_div_vals_by_corr_row_indices(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int fixed_row_gap_size);

    // 真正执行
    void run(bool check = true);

    unsigned long get_fixed_row_gap_size()
    {
        return this->fixed_row_gap_size;
    }

    int get_target_matrix_id()
    {
        return this->target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    unsigned long fixed_row_gap_size = 0;

    // 使用两个数组分别记录输入数据记录和输出数据记录
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

// 根据行索引来为行索引分块，因为是行索引分块，所有的行索引都应该变成相对索引
class fixed_div_row_indices : public basic_data_transform_step
{
public:
    fixed_div_row_indices(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int fixed_row_gap_size);

    // 真正执行
    void run(bool check = true);

    unsigned long get_fixed_row_gap_size()
    {
        return this->fixed_row_gap_size;
    }

    int get_target_matrix_id()
    {
        return this->target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    unsigned long fixed_row_gap_size = 0;

    // 使用两个数组分别记录输入数据记录和输出数据记录
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

// 增加行分割对于起始行记录修改
class modify_row_start_boundary_after_fixed_div_in_row_direction : public basic_data_transform_step
{
public:
    modify_row_start_boundary_after_fixed_div_in_row_direction(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int fixed_row_gap_size);

    // 真正执行
    void run(bool check = true);

    unsigned long get_fixed_row_gap_size()
    {
        return this->fixed_row_gap_size;
    }

    int get_target_matrix_id()
    {
        return this->target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    unsigned long fixed_row_gap_size = 0;

    // 使用两个数组分别记录输入数据记录和输出数据记录
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class modify_row_end_boundary_after_fixed_div_in_row_direction : public basic_data_transform_step
{
public:
    modify_row_end_boundary_after_fixed_div_in_row_direction(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int fixed_row_gap_size);

    // 真正执行
    void run(bool check = true);

    unsigned long get_fixed_row_gap_size()
    {
        return this->fixed_row_gap_size;
    }

    int get_target_matrix_id()
    {
        return this->target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    unsigned long fixed_row_gap_size = 0;

    // 使用两个数组分别记录输入数据记录和输出数据记录
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

// 根据矩阵行切分的结果，将母矩阵的列边界拷贝给非空子矩阵。列起始边界，只需要原封不动值拷贝即可
class modify_col_start_boundary_after_fixed_div_in_row_direction : public basic_data_transform_step
{
public:
    modify_col_start_boundary_after_fixed_div_in_row_direction(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int fixed_row_gap_size);

    // 真正执行
    void run(bool check = true);

    unsigned long get_fixed_row_gap_size()
    {
        return this->fixed_row_gap_size;
    }

    int get_target_matrix_id()
    {
        return this->target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    unsigned long fixed_row_gap_size = 0;

    // 使用两个数组分别记录输入数据记录和输出数据记录
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

// 根据矩阵行切分的结果，将母矩阵的列边界拷贝给非空子矩阵。列结束边界，只需要原封不动值拷贝即可
class modify_col_end_boundary_after_fixed_div_in_row_direction : public basic_data_transform_step
{
public:
    modify_col_end_boundary_after_fixed_div_in_row_direction(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int fixed_row_gap_size);

    // 真正执行
    void run(bool check = true);

    unsigned long get_fixed_row_gap_size()
    {
        return this->fixed_row_gap_size;
    }

    int get_target_matrix_id()
    {
        return this->target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    unsigned long fixed_row_gap_size = 0;

    // 使用两个数组分别记录输入数据记录和输出数据记录
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class get_begin_rows_of_BMTBs_after_fixed_blocking_in_row_direction : public basic_data_transform_step
{
public:
    get_begin_rows_of_BMTBs_after_fixed_blocking_in_row_direction(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int fixed_row_block_size);

    // 执行
    void run(bool check = true);

    // 被分块的目标子矩阵
    int get_fixed_row_block_size()
    {
        return fixed_row_block_size;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int fixed_row_block_size = 0;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class get_begin_nzs_of_BMTBs_after_fixed_blocking_in_row_direction : public basic_data_transform_step
{
public:
    get_begin_nzs_of_BMTBs_after_fixed_blocking_in_row_direction(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int fixed_row_block_size);

    // 执行
    void run(bool check = true);

    // 被分块的目标子矩阵
    int get_fixed_row_block_size()
    {
        return fixed_row_block_size;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int fixed_row_block_size = 0;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

// 去除一个子块末尾的空行，在排序中使用，通过这种方式可以去除空行
class remove_empty_row_in_end_of_sub_matrix : public basic_data_transform_step
{
public:
    remove_empty_row_in_end_of_sub_matrix(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id);

    // 执行
    void run(bool check = true);

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;

    // 使用两个数组分别记录输入数据记录和输出数据记录
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

// 对某个子矩阵在行方向上处理增加假的行，将行的数量添加到某个数的倍数，这些假的行不参与归约过程。当然在列方向也会有padding
class modify_row_indices_by_row_pad_in_sub_matrix : public basic_data_transform_step
{
public:
    modify_row_indices_by_row_pad_in_sub_matrix(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int multiple);

    // 执行
    void run(bool check = true);

    // 被分块的目标子矩阵
    int get_multiple()
    {
        return multiple;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int multiple = 0;

    // 需要动态决定，如果一开始的行数量就满足要求，那就不需要padding了
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class modify_col_indices_by_row_pad_in_sub_matrix : public basic_data_transform_step
{
public:
    modify_col_indices_by_row_pad_in_sub_matrix(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int multiple);

    // 执行
    void run(bool check = true);

    // 被分块的目标子矩阵
    int get_multiple()
    {
        return multiple;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int multiple = 0;

    // 需要动态决定，如果一开始的行数量就满足要求，那就不需要padding了
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class modify_vals_by_row_pad_in_sub_matrix : public basic_data_transform_step
{
public:
    modify_vals_by_row_pad_in_sub_matrix(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int multiple);

    // 执行
    void run(bool check = true);

    // 被分块的目标子矩阵
    int get_multiple()
    {
        return multiple;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int multiple = 0;

    // 需要动态决定，如果一开始的行数量就满足要求，那就不需要padding了
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

// 执行warp级别的切分，并且没有tblock级别的切分
class get_begin_rows_of_BMW_after_fixed_blocking_in_row_direction_without_BMTB : public basic_data_transform_step
{
public:
    get_begin_rows_of_BMW_after_fixed_blocking_in_row_direction_without_BMTB(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int fixed_row_block_size);

    // 执行
    void run(bool check = true);

    // 被分块的目标子矩阵
    int get_fixed_row_block_size()
    {
        return fixed_row_block_size;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int fixed_row_block_size = 0;

    // 需要动态决定，如果一开始的行数量就满足要求，那就不需要padding了
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

// 执行warp级别的切分，并且要BMTB的基础上进行切分
class get_begin_nzs_of_BMW_after_fixed_blocking_in_row_direction_without_BMTB : public basic_data_transform_step
{
public:
    get_begin_nzs_of_BMW_after_fixed_blocking_in_row_direction_without_BMTB(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int fixed_row_block_size);

    // 执行
    void run(bool check = true);

    // 被分块的目标子矩阵
    int get_fixed_row_block_size()
    {
        return fixed_row_block_size;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int fixed_row_block_size = 0;

    // 需要动态决定，如果一开始的行数量就满足要求，那就不需要padding了
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

// 在BMTB切分的基础上执行BMW行切分，warp的起始行是全局索引
class get_begin_rows_of_BMW_after_fixed_blocking_in_row_direction_in_BMTB : public basic_data_transform_step
{
public:
    // 在BMTB中执行分块，并映射到BMW
    get_begin_rows_of_BMW_after_fixed_blocking_in_row_direction_in_BMTB(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int fixed_row_block_size);

    // 执行
    void run(bool check = true);

    // 被分块的目标子矩阵
    int get_fixed_row_block_size()
    {
        return fixed_row_block_size;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int fixed_row_block_size = 0;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

// 在BMTB切分的基础上执行BMW行切分，warp的起始行是相对索引
class get_begin_rows_of_BMW_after_fixed_blocking_in_row_direction_relative_to_BMTB : public basic_data_transform_step
{
public:
    get_begin_rows_of_BMW_after_fixed_blocking_in_row_direction_relative_to_BMTB(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int fixed_row_block_size);

    // 执行
    void run(bool check = true);

    // 被分块的目标子矩阵
    int get_fixed_row_block_size()
    {
        return fixed_row_block_size;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int fixed_row_block_size = 0;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

// 在BMTB切分的基础上执行BMW行切分，warp的起始非零元是全局的
class get_begin_nzs_of_BMW_after_fixed_blocking_in_row_direction_in_BMTB : public basic_data_transform_step
{
public:
    get_begin_nzs_of_BMW_after_fixed_blocking_in_row_direction_in_BMTB(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int fixed_row_block_size);

    // 执行
    void run(bool check = true);

    // 被分块的目标子矩阵
    int get_fixed_row_block_size()
    {
        return fixed_row_block_size;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int fixed_row_block_size = 0;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

// 在BMTB切分的基础上执行BMW行切分，warp的起始非零元是相对的
class get_begin_nzs_of_BMW_after_fixed_blocking_in_row_direction_relative_to_BMTB : public basic_data_transform_step
{
public:
    get_begin_nzs_of_BMW_after_fixed_blocking_in_row_direction_relative_to_BMTB(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int fixed_row_block_size);

    void run(bool check = true);

    int get_fixed_row_block_size()
    {
        return fixed_row_block_size;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int fixed_row_block_size = 0;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

// 在BMW在BMTB内切分时，获得每一个BMTB相对于BMW的偏移量
class get_begin_BMWs_of_BMTB_after_blocking_in_row_direction : public basic_data_transform_step
{
public:
    get_begin_BMWs_of_BMTB_after_blocking_in_row_direction(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id);

    void run(bool check = true);

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

// 在一开始的时候就将每一行的非零元数量padding到固定的倍数，这是针对全局的padding此外还有针对每一个BMTB、BMW内部的padding，这种padding一般出现在BMT的单行切分上
class modify_row_indices_by_col_pad_in_sub_matrix : public basic_data_transform_step
{
public:
    modify_row_indices_by_col_pad_in_sub_matrix(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int multiple_of_each_row_size);

    void run(bool check = true);

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    int get_multiple_of_each_row_size()
    {
        return multiple_of_each_row_size;
    }

    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int multiple_of_each_row_size = 0;

    // 如果一开始就满足了padding之后的要求，那就不需要是
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

// 将一个子矩阵的每一行padding到对应倍数的长度，记录列索引的变化
class modify_col_indices_by_col_pad_in_sub_matrix : public basic_data_transform_step
{
public:
    modify_col_indices_by_col_pad_in_sub_matrix(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int multiple_of_each_row_size);

    void run(bool check = true);

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    int get_multiple_of_each_row_size()
    {
        return multiple_of_each_row_size;
    }

    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int multiple_of_each_row_size = 0;

    // 如果一开始就满足了padding之后的要求，那就不需要是
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class modify_vals_by_col_pad_in_sub_matrix : public basic_data_transform_step
{
public:
    modify_vals_by_col_pad_in_sub_matrix(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int multiple_of_each_row_size);

    void run(bool check = true);

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    int get_multiple_of_each_row_size()
    {
        return multiple_of_each_row_size;
    }

    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int multiple_of_each_row_size = 0;

    // 如果一开始就满足了padding之后的要求，那就不需要是
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

// BMT列切分，获得每一个BMT的首行索引，这个首行索引是不带尾巴的（最后不加一个总行数，索引和数量和BMT数量一致），在父块的基础上做列切分，得到每一个BMT的行偏移量
class get_begin_rows_of_BMT_after_fixed_blocking_in_col_direction : public basic_data_transform_step
{
public:
    get_begin_rows_of_BMT_after_fixed_blocking_in_col_direction(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int col_size);

    void run(bool check = true);

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    int get_col_size()
    {
        return col_size;
    }

    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int col_size = 0;

    // 如果一开始就满足了padding之后的要求，那就不需要是
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

// 获得BMT相对于BMTB的行索引
class get_begin_rows_of_BMT_after_fixed_blocking_in_col_direction_relative_to_BMTB : public basic_data_transform_step
{
public:
    get_begin_rows_of_BMT_after_fixed_blocking_in_col_direction_relative_to_BMTB(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int col_size);

    void run(bool check = true);

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    int get_col_size()
    {
        return col_size;
    }

    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int col_size = 0;

    // 如果一开始就满足了padding之后的要求，那就不需要是
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

// 获得BMT相对于BMW的行索引
class get_begin_rows_of_BMT_after_fixed_blocking_in_col_direction_relative_to_BMW : public basic_data_transform_step
{
public:
    get_begin_rows_of_BMT_after_fixed_blocking_in_col_direction_relative_to_BMW(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int col_size);

    void run(bool check = true);

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    int get_col_size()
    {
        return col_size;
    }

    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int col_size = 0;

    // 如果一开始就满足了padding之后的要求，那就不需要是
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

// BMT列切分，获得每一个BMT的首nz索引，这个首nz索引是带尾巴的，是直接在原始的子块中做切分
// 即便存在父块，也不影响分块结果。
class get_begin_nzs_of_BMT_after_fixed_blocking_in_col_direction : public basic_data_transform_step
{
public:
    get_begin_nzs_of_BMT_after_fixed_blocking_in_col_direction(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int col_size);

    void run(bool check = true);

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    int get_col_size()
    {
        return col_size;
    }

    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int col_size = 0;

    // 如果一开始就满足了padding之后的要求，那就不需要是
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

// BMT的列切分，根据传入的参数，决定是相对于哪个父块
class get_begin_nzs_of_BMT_after_fixed_blocking_in_col_direction_relative_to_parents : public basic_data_transform_step
{
public:
    get_begin_nzs_of_BMT_after_fixed_blocking_in_col_direction_relative_to_parents(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int col_size, POS_TYPE parent_pos);

    void run(bool check = true);

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    int get_col_size()
    {
        return col_size;
    }

    POS_TYPE get_parent_pos()
    {
        return parent_pos;
    }

    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int col_size = 0;
    POS_TYPE parent_pos = NONE_META;

    // 如果一开始就满足了padding之后的要求，那就不需要是
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

// 如果是padding过的切分，可以存储父块中每一个BMT的大小，由此函数内部判断BMT大小是否一致，不一致则return， is_run为FALSE， 置source 和 dest 为空
class get_BMT_size_of_each_parent : public basic_data_transform_step
{
public:
    get_BMT_size_of_each_parent(shared_ptr<meta_data_set> meta_data_set_ptr, POS_TYPE parent_pos, int target_matrix_id, bool row_direction_blocking);

    void run(bool check = true);

    POS_TYPE get_parent_pos()
    {
        return parent_pos;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    string convert_to_string();

    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

private:
    POS_TYPE parent_pos = NONE_META;
    int target_matrix_id = -1;
    // 如果get_BMT是在行切分中使用，则需要在父块为空时，仍然计算BMT id
    bool row_direction_blocking = false;
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

// 获得每一个父块中的BMT的偏移量，使用非零元偏移量来执行计算，这不是最直接的方法，但是做起来比较简单
class get_begin_BMTs_of_specific_parent_after_blocking : public basic_data_transform_step
{
public:
    get_begin_BMTs_of_specific_parent_after_blocking(shared_ptr<meta_data_set> meta_data_set_ptr, POS_TYPE parent_pos, int target_matrix_id);

    void run(bool check = true);

    POS_TYPE get_parent_pos()
    {
        return parent_pos;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    string convert_to_string();

    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

private:
    POS_TYPE parent_pos = NONE_META;
    int target_matrix_id = -1;

    // 如果一开始就满足了padding之后的要求，那就不需要是
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

// BMT相对于某一个父块的padding。使得每一个父块中的行非零元都padding到最大行长度的大小，其可能会导致一系列的重算，因为会改变BMT和BMTB的分块索引
class modify_col_indices_by_col_pad_parent_blk_to_max_row_size : public basic_data_transform_step
{
public:
    // 父块可以是TBLOCK、WARP和GLOBAL级别的
    modify_col_indices_by_col_pad_parent_blk_to_max_row_size(shared_ptr<meta_data_set> meta_data_set_ptr, POS_TYPE parent_pos, int target_matrix_id, bool padding_with_empty_row);

    void run(bool check = true);

    POS_TYPE get_parent_pos()
    {
        return parent_pos;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    string convert_to_string();

    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

private:
    POS_TYPE parent_pos = NONE_META;
    int target_matrix_id = -1;
    bool padding_with_empty_row = false;
    // 如果一开始就满足了padding之后的要求，那就不需要是
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class modify_vals_by_col_pad_parent_blk_to_max_row_size : public basic_data_transform_step
{
public:
    modify_vals_by_col_pad_parent_blk_to_max_row_size(shared_ptr<meta_data_set> meta_data_set_ptr, POS_TYPE parent_pos, int target_matrix_id, bool padding_with_empty_row);

    void run(bool check = true);

    POS_TYPE get_parent_pos()
    {
        return parent_pos;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    string convert_to_string();

    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

private:
    POS_TYPE parent_pos = NONE_META;
    int target_matrix_id = -1;
    bool padding_with_empty_row = false;
    // 如果一开始就满足了padding之后的要求，那就不需要是
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class modify_row_indices_by_col_pad_parent_blk_to_max_row_size : public basic_data_transform_step
{
public:
    modify_row_indices_by_col_pad_parent_blk_to_max_row_size(shared_ptr<meta_data_set> meta_data_set_ptr, POS_TYPE parent_pos, int target_matrix_id, bool padding_with_empty_row);

    void run(bool check = true);

    POS_TYPE get_parent_pos()
    {
        return parent_pos;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    string convert_to_string();

    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

private:
    POS_TYPE parent_pos = NONE_META;
    int target_matrix_id = -1;
    bool padding_with_empty_row = false;
    // 如果一开始就满足了padding之后的要求，那就不需要是
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class modify_col_indices_by_empty_pad_in_submatrix : public basic_data_transform_step
{
public:
    modify_col_indices_by_empty_pad_in_submatrix(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id);

    void run(bool check = true);

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    string convert_to_string();

    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

private:
    int target_matrix_id = -1;

    // 如果一开始就满足了padding之后的要求，那就不需要是
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class modify_vals_by_empty_pad_in_submatrix : public basic_data_transform_step
{
public:
    modify_vals_by_empty_pad_in_submatrix(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id);

    void run(bool check = true);

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    string convert_to_string();

    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

private:
    int target_matrix_id = -1;

    // 如果一开始就满足了padding之后的要求，那就不需要是
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class modify_row_indices_by_empty_pad_in_submatrix : public basic_data_transform_step
{
public:
    modify_row_indices_by_empty_pad_in_submatrix(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id);

    void run(bool check = true);

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    string convert_to_string();

    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

private:
    int target_matrix_id = -1;

    // 如果一开始就满足了padding之后的要求，那就不需要是
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class remove_item_of_metadata : public basic_data_transform_step
{
public:
    remove_item_of_metadata(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, string item_name, POS_TYPE pos);

    void run(bool check = true);

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    string get_item_name()
    {
        return item_name;
    }

    POS_TYPE get_pos()
    {
        return pos;
    }

    string convert_to_string();

    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

private:
    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();

    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    int target_matrix_id = -1;
    string item_name;
    POS_TYPE pos;
};

class modify_row_indices_by_interlance_storage : public basic_data_transform_step
{
public:
    modify_row_indices_by_interlance_storage(shared_ptr<meta_data_set> meta_data_set_ptr, POS_TYPE parent_pos, int target_matrix_id);

    void run(bool check = true);

    POS_TYPE get_parent_pos()
    {
        return parent_pos;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    string convert_to_string();

    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

private:
    POS_TYPE parent_pos = NONE_META;
    int target_matrix_id = -1;

    // 如果一开始就满足了padding之后的要求，那就不需要是
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class modify_col_indices_by_interlance_storage : public basic_data_transform_step
{
public:
    modify_col_indices_by_interlance_storage(shared_ptr<meta_data_set> meta_data_set_ptr, POS_TYPE parent_pos, int target_matrix_id);

    void run(bool check = true);

    POS_TYPE get_parent_pos()
    {
        return parent_pos;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    string convert_to_string();

    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

private:
    POS_TYPE parent_pos = NONE_META;
    int target_matrix_id = -1;

    // 如果一开始就满足了padding之后的要求，那就不需要是
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class modify_vals_by_interlance_storage : public basic_data_transform_step
{
public:
    modify_vals_by_interlance_storage(shared_ptr<meta_data_set> meta_data_set_ptr, POS_TYPE parent_pos, int target_matrix_id);

    void run(bool check = true);

    POS_TYPE get_parent_pos()
    {
        return parent_pos;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    string convert_to_string();

    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

private:
    POS_TYPE parent_pos = NONE_META;
    int target_matrix_id = -1;

    // 如果一开始就满足了padding之后的要求，那就不需要是
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

// 执行thread级别的切分
class get_begin_rows_of_BMT_after_fixed_blocking_in_row_direction : public basic_data_transform_step
{
public:
    get_begin_rows_of_BMT_after_fixed_blocking_in_row_direction(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int fixed_row_block_size);

    // 执行
    void run(bool check = true);

    // 被分块的目标子矩阵
    int get_fixed_row_block_size()
    {
        return fixed_row_block_size;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int fixed_row_block_size = 0;

    // 需要动态决定，如果一开始的行数量就满足要求，那就不需要padding了
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class get_begin_nzs_of_BMT_after_fixed_blocking_in_row_direction : public basic_data_transform_step
{
public:
    get_begin_nzs_of_BMT_after_fixed_blocking_in_row_direction(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int fixed_row_block_size);

    // 执行
    void run(bool check = true);

    // 被分块的目标子矩阵
    int get_fixed_row_block_size()
    {
        return fixed_row_block_size;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int fixed_row_block_size = 0;

    // 需要动态决定，如果一开始的行数量就满足要求，那就不需要padding了
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class get_begin_rows_of_BMT_after_fixed_blocking_in_row_direction_in_BMTB : public basic_data_transform_step
{
public:
    get_begin_rows_of_BMT_after_fixed_blocking_in_row_direction_in_BMTB(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int fixed_row_block_size);

    // 执行
    void run(bool check = true);

    // 被分块的目标子矩阵
    int get_fixed_row_block_size()
    {
        return fixed_row_block_size;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int fixed_row_block_size = 0;

    // 需要动态决定，如果一开始的行数量就满足要求，那就不需要padding了
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class get_begin_nzs_of_BMT_after_fixed_blocking_in_row_direction_in_BMTB : public basic_data_transform_step
{
public:
    get_begin_nzs_of_BMT_after_fixed_blocking_in_row_direction_in_BMTB(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int fixed_row_block_size);

    // 执行
    void run(bool check = true);

    // 被分块的目标子矩阵
    int get_fixed_row_block_size()
    {
        return fixed_row_block_size;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int fixed_row_block_size = 0;

    // 需要动态决定，如果一开始的行数量就满足要求，那就不需要padding了
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class get_begin_rows_of_BMT_after_fixed_blocking_in_row_direction_in_BMW : public basic_data_transform_step
{
public:
    get_begin_rows_of_BMT_after_fixed_blocking_in_row_direction_in_BMW(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int fixed_row_block_size);

    // 执行
    void run(bool check = true);

    // 被分块的目标子矩阵
    int get_fixed_row_block_size()
    {
        return fixed_row_block_size;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int fixed_row_block_size = 0;

    // 需要动态决定，如果一开始的行数量就满足要求，那就不需要padding了
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class get_begin_nzs_of_BMT_after_fixed_blocking_in_row_direction_in_BMW : public basic_data_transform_step
{
public:
    get_begin_nzs_of_BMT_after_fixed_blocking_in_row_direction_in_BMW(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int fixed_row_block_size);

    // 执行
    void run(bool check = true);

    // 被分块的目标子矩阵
    int get_fixed_row_block_size()
    {
        return fixed_row_block_size;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int fixed_row_block_size = 0;

    // 需要动态决定，如果一开始的行数量就满足要求，那就不需要padding了
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class get_begin_rows_of_BMT_after_fixed_blocking_in_row_direction_relative_to_BMTB : public basic_data_transform_step
{
public:
    get_begin_rows_of_BMT_after_fixed_blocking_in_row_direction_relative_to_BMTB(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int fixed_row_block_size);

    void run(bool check = true);

    int get_fixed_row_block_size()
    {
        return fixed_row_block_size;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int fixed_row_block_size = 0;
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class get_begin_nzs_of_BMT_after_fixed_blocking_in_row_direction_relative_to_BMTB : public basic_data_transform_step
{
public:
    get_begin_nzs_of_BMT_after_fixed_blocking_in_row_direction_relative_to_BMTB(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int fixed_row_block_size);

    void run(bool check = true);

    int get_fixed_row_block_size()
    {
        return fixed_row_block_size;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int fixed_row_block_size = 0;
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class get_begin_rows_of_BMT_after_fixed_blocking_in_row_direction_relative_to_BMW : public basic_data_transform_step
{
public:
    get_begin_rows_of_BMT_after_fixed_blocking_in_row_direction_relative_to_BMW(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int fixed_row_block_size);

    void run(bool check = true);

    int get_fixed_row_block_size()
    {
        return fixed_row_block_size;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int fixed_row_block_size = 0;
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class get_begin_nzs_of_BMT_after_fixed_blocking_in_row_direction_relative_to_BMW : public basic_data_transform_step
{
public:
    get_begin_nzs_of_BMT_after_fixed_blocking_in_row_direction_relative_to_BMW(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int fixed_row_block_size);

    void run(bool check = true);

    int get_fixed_row_block_size()
    {
        return fixed_row_block_size;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int fixed_row_block_size = 0;
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

// 获得每一个父块中的BMT的偏移量，使用非零元偏移量来执行计算，这不是最直接的方法，但是做起来比较简单
class get_begin_BMTs_of_specific_parent_after_blocking_in_row_direction : public basic_data_transform_step
{
public:
    get_begin_BMTs_of_specific_parent_after_blocking_in_row_direction(shared_ptr<meta_data_set> meta_data_set_ptr, POS_TYPE parent_pos, int target_matrix_id);

    void run(bool check = true);

    POS_TYPE get_parent_pos()
    {
        return parent_pos;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    string convert_to_string();

    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

private:
    POS_TYPE parent_pos = NONE_META;
    int target_matrix_id = -1;

    // 如果一开始就满足了padding之后的要求，那就不需要是
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class get_begin_rows_of_BMW_after_fixed_blocking_in_col_direction : public basic_data_transform_step
{
public:
    get_begin_rows_of_BMW_after_fixed_blocking_in_col_direction(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int col_size);

    void run(bool check = true);

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    int get_col_size()
    {
        return fixed_col_block_size;
    }

    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int fixed_col_block_size = 0;

    // 如果一开始就满足了padding之后的要求，那就不需要是
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class get_begin_nzs_of_BMW_after_fixed_blocking_in_col_direction : public basic_data_transform_step
{
public:
    get_begin_nzs_of_BMW_after_fixed_blocking_in_col_direction(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int col_size);

    void run(bool check = true);

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    int get_col_size()
    {
        return fixed_col_block_size;
    }

    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int fixed_col_block_size = 0;

    // 如果一开始就满足了padding之后的要求，那就不需要是
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class get_begin_rows_of_BMW_after_fixed_blocking_in_col_direction_relative_to_BMTB : public basic_data_transform_step
{
public:
    get_begin_rows_of_BMW_after_fixed_blocking_in_col_direction_relative_to_BMTB(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int fixed_col_block_size);

    void run(bool check = true);

    int get_fixed_col_block_size()
    {
        return fixed_col_block_size;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int fixed_col_block_size = 0;
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class get_begin_nzs_of_BMW_after_fixed_blocking_in_col_direction_relative_to_BMTB : public basic_data_transform_step
{
public:
    get_begin_nzs_of_BMW_after_fixed_blocking_in_col_direction_relative_to_BMTB(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int fixed_col_block_size);

    void run(bool check = true);

    int get_fixed_col_block_size()
    {
        return fixed_col_block_size;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int fixed_col_block_size = 0;
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class get_begin_BMWs_of_BMTB_after_blocking : public basic_data_transform_step
{
public:
    get_begin_BMWs_of_BMTB_after_blocking(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id);

    void run(bool check = true);

    int get_fixed_col_block_size()
    {
        return fixed_col_block_size;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int fixed_col_block_size = 0;
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class get_BMW_size_of_each_parent : public basic_data_transform_step
{
public:
    get_BMW_size_of_each_parent(shared_ptr<meta_data_set> meta_data_set_ptr, POS_TYPE parent_pos, int target_matrix_id);

    void run(bool check = true);

    POS_TYPE get_parent_pos()
    {
        return parent_pos;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    string convert_to_string();

    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

private:
    POS_TYPE parent_pos = NONE_META;
    int target_matrix_id = -1;

    // 如果一开始就满足了padding之后的要求，那就不需要是
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class get_BMTB_size : public basic_data_transform_step
{
public:
    get_BMTB_size(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id);

    void run(bool check = true);

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    string convert_to_string();

    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

private:
    int target_matrix_id = -1;

    // 如果一开始就满足了padding之后的要求，那就不需要是
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class get_begin_rows_of_BMTB_after_fixed_blocking_in_col_direction : public basic_data_transform_step
{
public:
    get_begin_rows_of_BMTB_after_fixed_blocking_in_col_direction(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int col_size);

    void run(bool check = true);

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    int get_col_size()
    {
        return fixed_col_block_size;
    }

    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int fixed_col_block_size = 0;

    // 如果一开始就满足了padding之后的要求，那就不需要是
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class get_begin_nzs_of_BMTB_after_fixed_blocking_in_col_direction : public basic_data_transform_step
{
public:
    get_begin_nzs_of_BMTB_after_fixed_blocking_in_col_direction(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int col_size);

    void run(bool check = true);

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    int get_col_size()
    {
        return fixed_col_block_size;
    }

    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int fixed_col_block_size = 0;

    // 如果一开始就满足了padding之后的要求，那就不需要是
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class modify_row_start_boundary_after_div_according_to_row_nz : public basic_data_transform_step
{
public:
    modify_row_start_boundary_after_div_according_to_row_nz(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int nz_gap_size, int max_gap, int expansion_rate);

    // 真正执行
    void run(bool check = true);

    int get_nz_gap_size()
    {
        return this->nz_gap_size;
    }

    int get_expansion_rate()
    {
        return this->expansion_rate;
    }

    int get_target_matrix_id()
    {
        return this->target_matrix_id;
    }

    int get_max_gap()
    {
        return this->max_gap;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int nz_gap_size = 0;
    int max_gap = 0;
    int expansion_rate = 0;

    // 使用两个数组分别记录输入数据记录和输出数据记录
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class modify_row_end_boundary_after_div_according_to_row_nz : public basic_data_transform_step
{
public:
    modify_row_end_boundary_after_div_according_to_row_nz(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int nz_gap_size, int max_gap, int expansion_rate);

    // 真正执行
    void run(bool check = true);

    int get_nz_gap_size()
    {
        return this->nz_gap_size;
    }

    int get_expansion_rate()
    {
        return this->expansion_rate;
    }

    int get_max_gap()
    {
        return this->max_gap;
    }

    int get_target_matrix_id()
    {
        return this->target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int nz_gap_size = 0;
    int max_gap = 0;
    int expansion_rate = 0;

    // 使用两个数组分别记录输入数据记录和输出数据记录
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class modify_col_start_boundary_after_div_according_to_row_nz : public basic_data_transform_step
{
public:
    modify_col_start_boundary_after_div_according_to_row_nz(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int nz_gap_size, int max_gap, int expansion_rate);

    // 真正执行
    void run(bool check = true);

    int get_nz_gap_size()
    {
        return this->nz_gap_size;
    }
    int get_max_gap()
    {
        return this->max_gap;
    }

    int get_expansion_rate()
    {
        return this->expansion_rate;
    }

    int get_target_matrix_id()
    {
        return this->target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int nz_gap_size = 0;
    int max_gap = 0;
    int expansion_rate = 0;

    // 使用两个数组分别记录输入数据记录和输出数据记录
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class modify_col_end_boundary_after_div_according_to_row_nz : public basic_data_transform_step
{
public:
    modify_col_end_boundary_after_div_according_to_row_nz(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int nz_gap_size, int max_gap, int expansion_rate);

    // 真正执行
    void run(bool check = true);

    int get_nz_gap_size()
    {
        return this->nz_gap_size;
    }

    int get_max_gap()
    {
        return this->max_gap;
    }

    int get_expansion_rate()
    {
        return this->expansion_rate;
    }

    int get_target_matrix_id()
    {
        return this->target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int nz_gap_size = 0;
    int max_gap = 0;
    int expansion_rate = 0;

    // 使用两个数组分别记录输入数据记录和输出数据记录
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class div_col_indices_by_row_nnz : public basic_data_transform_step
{
public:
    div_col_indices_by_row_nnz(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int nz_gap_size, int max_gap, int expansion_rate);

    // 真正执行
    void run(bool check = true);

    int get_nz_gap_size()
    {
        return this->nz_gap_size;
    }

    int get_max_gap()
    {
        return this->max_gap;
    }

    int get_expansion_rate()
    {
        return this->expansion_rate;
    }

    int get_target_matrix_id()
    {
        return this->target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int nz_gap_size = 0;
    int max_gap = 0;
    int expansion_rate = 0;

    // 使用两个数组分别记录输入数据记录和输出数据记录
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class div_val_indices_by_row_nnz : public basic_data_transform_step
{
public:
    div_val_indices_by_row_nnz(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int nz_gap_size, int max_gap, int expansion_rate);

    // 真正执行
    void run(bool check = true);

    int get_nz_gap_size()
    {
        return this->nz_gap_size;
    }

    int get_max_gap()
    {
        return this->max_gap;
    }

    int get_expansion_rate()
    {
        return this->expansion_rate;
    }

    int get_target_matrix_id()
    {
        return this->target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int nz_gap_size = 0;
    int max_gap = 0;
    int expansion_rate = 0;

    // 使用两个数组分别记录输入数据记录和输出数据记录
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class div_row_indices_by_row_nnz : public basic_data_transform_step
{
public:
    div_row_indices_by_row_nnz(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int nz_gap_size, int max_gap, int expansion_rate);

    // 真正执行
    void run(bool check = true);

    int get_nz_gap_size()
    {
        return this->nz_gap_size;
    }

    int get_expansion_rate()
    {
        return this->expansion_rate;
    }

    int get_max_gap()
    {
        return this->max_gap;
    }

    int get_target_matrix_id()
    {
        return this->target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int nz_gap_size = 0;
    int max_gap = 0;
    int expansion_rate = 0;

    // 使用两个数组分别记录输入数据记录和输出数据记录
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class modify_indices_by_col_pad_parent_blk_to_max_row_size : public merge_data_transform_step
{
public:
    modify_indices_by_col_pad_parent_blk_to_max_row_size(shared_ptr<meta_data_set> meta_data_set_ptr, POS_TYPE parent_pos, int target_matrix_id, bool row_flag, bool col_flag, bool val_flag, int fixed_row_block_size = 1);

    void run(bool check = true);

    POS_TYPE get_parent_pos()
    {
        return parent_pos;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    string convert_to_string();

    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

private:
    POS_TYPE parent_pos = NONE_META;
    int target_matrix_id = -1;
    bool row_flag = true;
    bool col_flag = true;
    bool val_flag = true;
    int fixed_row_block_size = 0;

    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class modify_col_indices_by_col_pad_parent_blk_to_max_row_size_new : public merge_data_transform_step
{
public:
    modify_col_indices_by_col_pad_parent_blk_to_max_row_size_new(shared_ptr<meta_data_set> meta_data_set_ptr, POS_TYPE parent_pos, int target_matrix_id);

    void run(bool check = true);

    POS_TYPE get_parent_pos()
    {
        return parent_pos;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    string convert_to_string();

    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

private:
    POS_TYPE parent_pos = NONE_META;
    int target_matrix_id = -1;
    bool row_flag = true;
    bool col_flag = true;
    bool val_flag = true;
    int fixed_row_block_size = 0;

    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class modify_row_indices_by_col_pad_parent_blk_to_max_row_size_new : public merge_data_transform_step
{
public:
    modify_row_indices_by_col_pad_parent_blk_to_max_row_size_new(shared_ptr<meta_data_set> meta_data_set_ptr, POS_TYPE parent_pos, int target_matrix_id);

    void run(bool check = true);

    POS_TYPE get_parent_pos()
    {
        return parent_pos;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    string convert_to_string();

    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

private:
    POS_TYPE parent_pos = NONE_META;
    int target_matrix_id = -1;
    bool row_flag = true;
    bool col_flag = true;
    bool val_flag = true;
    int fixed_row_block_size = 0;

    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class modify_vals_by_col_pad_parent_blk_to_max_row_size_new : public merge_data_transform_step
{
public:
    modify_vals_by_col_pad_parent_blk_to_max_row_size_new(shared_ptr<meta_data_set> meta_data_set_ptr, POS_TYPE parent_pos, int target_matrix_id);

    void run(bool check = true);

    POS_TYPE get_parent_pos()
    {
        return parent_pos;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    string convert_to_string();

    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

private:
    POS_TYPE parent_pos = NONE_META;
    int target_matrix_id = -1;
    bool row_flag = true;
    bool col_flag = true;
    bool val_flag = true;
    int fixed_row_block_size = 0;

    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};
class get_begin_nzs_of_BMT_after_fixed_blocking_in_row_direction_new : public merge_data_transform_step
{
public:
    get_begin_nzs_of_BMT_after_fixed_blocking_in_row_direction_new(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int fixed_row_block_size);

    // 执行
    void run(bool check = true);

    int get_fixed_row_block_size()
    {
        return fixed_row_block_size;
    }

    // 被分块的目标子矩阵
    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int fixed_row_block_size = 0;

    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class get_BMT_size_of_each_parent_new : public merge_data_transform_step
{
public:
    get_BMT_size_of_each_parent_new(shared_ptr<meta_data_set> meta_data_set_ptr, POS_TYPE parent_pos, int target_matrix_id);

    void run(bool check = true);

    POS_TYPE get_parent_pos()
    {
        return parent_pos;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    string convert_to_string();

    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

private:
    POS_TYPE parent_pos = NONE_META;
    int target_matrix_id = -1;

    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class get_begin_rows_of_BMT_after_nnz_blocking_in_row_direction_in_BMW : public basic_data_transform_step
{
public:
    get_begin_rows_of_BMT_after_nnz_blocking_in_row_direction_in_BMW(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int nnz_per_interval);

    // 执行
    void run(bool check = true);

    int get_nnz_per_interval()
    {
        return nnz_per_interval;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int nnz_per_interval = 0;

    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class get_begin_rows_of_BMT_after_nnz_blocking_in_row_direction_relative_to_BMW : public basic_data_transform_step
{
public:
    get_begin_rows_of_BMT_after_nnz_blocking_in_row_direction_relative_to_BMW(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int nnz_per_interval);

    void run(bool check = true);

    int get_nnz_per_interval()
    {
        return nnz_per_interval;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int nnz_per_interval = 0;
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class get_begin_rows_of_BMT_after_nnz_blocking_in_row_direction_relative_to_BMTB : public basic_data_transform_step
{
public:
    get_begin_rows_of_BMT_after_nnz_blocking_in_row_direction_relative_to_BMTB(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int nnz_per_interval);

    void run(bool check = true);

    int get_nnz_per_interval()
    {
        return nnz_per_interval;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int nnz_per_interval = 0;
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class get_begin_rows_of_BMT_after_nnz_blocking_in_row_direction_in_BMTB : public basic_data_transform_step
{
public:
    get_begin_rows_of_BMT_after_nnz_blocking_in_row_direction_in_BMTB(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int nnz_per_interval);

    void run(bool check = true);

    int get_nnz_per_interval()
    {
        return nnz_per_interval;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int nnz_per_interval = 0;
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class get_begin_rows_of_BMT_after_nnz_blocking_in_row_direction : public basic_data_transform_step
{
public:
    get_begin_rows_of_BMT_after_nnz_blocking_in_row_direction(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int nnz_per_interval);

    // 执行
    void run(bool check = true);

    int get_nnz_per_interval()
    {
        return nnz_per_interval;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int nnz_per_interval = 0;

    // 需要动态决定，如果一开始的行数量就满足要求，那就不需要padding了
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class get_begin_nzs_of_BMT_after_nnz_blocking_in_row_direction : public basic_data_transform_step
{
public:
    get_begin_nzs_of_BMT_after_nnz_blocking_in_row_direction(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int nnz_per_interval);

    // 执行
    void run(bool check = true);

    int get_nnz_per_interval()
    {
        return nnz_per_interval;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int nnz_per_interval = 0;

    // 需要动态决定，如果一开始的行数量就满足要求，那就不需要padding了
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class get_begin_nzs_of_BMT_after_nnz_blocking_in_row_direction_in_BMTB : public basic_data_transform_step
{
public:
    get_begin_nzs_of_BMT_after_nnz_blocking_in_row_direction_in_BMTB(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int nnz_per_interval);

    // 执行
    void run(bool check = true);

    int get_nnz_per_interval()
    {
        return nnz_per_interval;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int nnz_per_interval = 0;

    // 需要动态决定，如果一开始的行数量就满足要求，那就不需要padding了
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class get_begin_nzs_of_BMT_after_nnz_blocking_in_row_direction_in_BMW : public basic_data_transform_step
{
public:
    get_begin_nzs_of_BMT_after_nnz_blocking_in_row_direction_in_BMW(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int nnz_per_interval);

    // 执行
    void run(bool check = true);

    int get_nnz_per_interval()
    {
        return nnz_per_interval;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int nnz_per_interval = 0;

    // 需要动态决定，如果一开始的行数量就满足要求，那就不需要padding了
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class get_begin_nzs_of_BMT_after_nnz_blocking_in_row_direction_relative_to_BMTB : public basic_data_transform_step
{
public:
    get_begin_nzs_of_BMT_after_nnz_blocking_in_row_direction_relative_to_BMTB(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int nnz_per_interval);

    void run(bool check = true);

    int get_nnz_per_interval()
    {
        return nnz_per_interval;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int nnz_per_interval = 0;
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class get_begin_nzs_of_BMT_after_nnz_blocking_in_row_direction_relative_to_BMW : public basic_data_transform_step
{
public:
    get_begin_nzs_of_BMT_after_nnz_blocking_in_row_direction_relative_to_BMW(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int nnz_per_interval);

    void run(bool check = true);

    int get_nnz_per_interval()
    {
        return nnz_per_interval;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int nnz_per_interval = 0;
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class get_begin_nzs_of_BMW_after_nnz_blocking_in_row_direction_in_BMTB : public basic_data_transform_step
{
public:
    get_begin_nzs_of_BMW_after_nnz_blocking_in_row_direction_in_BMTB(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int nnz_per_interval);

    // 执行
    void run(bool check = true);

    int get_nnz_per_interval()
    {
        return nnz_per_interval;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int nnz_per_interval = 0;

    // 需要动态决定，如果一开始的行数量就满足要求，那就不需要padding了
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class get_begin_nzs_of_BMW_after_nnz_blocking_in_row_direction_relative_to_BMTB : public basic_data_transform_step
{
public:
    get_begin_nzs_of_BMW_after_nnz_blocking_in_row_direction_relative_to_BMTB(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int nnz_per_interval);

    // 执行
    void run(bool check = true);

    int get_nnz_per_interval()
    {
        return nnz_per_interval;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int nnz_per_interval = 0;

    // 需要动态决定，如果一开始的行数量就满足要求，那就不需要padding了
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class get_begin_rows_of_BMW_after_nnz_blocking_in_row_direction_in_BMTB : public basic_data_transform_step
{
public:
    get_begin_rows_of_BMW_after_nnz_blocking_in_row_direction_in_BMTB(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int nnz_per_interval);

    void run(bool check = true);

    int get_nnz_per_interval()
    {
        return nnz_per_interval;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int nnz_per_interval = 0;
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class get_begin_rows_of_BMW_after_nnz_blocking_in_row_direction_relative_to_BMTB : public basic_data_transform_step
{
public:
    get_begin_rows_of_BMW_after_nnz_blocking_in_row_direction_relative_to_BMTB(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int nnz_per_interval);

    void run(bool check = true);

    int get_nnz_per_interval()
    {
        return nnz_per_interval;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int nnz_per_interval = 0;
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class get_begin_nzs_of_BMW_after_nnz_blocking_in_row_direction : public basic_data_transform_step
{
public:
    get_begin_nzs_of_BMW_after_nnz_blocking_in_row_direction(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int nnz_per_interval);

    // 执行
    void run(bool check = true);

    int get_nnz_per_interval()
    {
        return nnz_per_interval;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int nnz_per_interval = 0;

    // 需要动态决定，如果一开始的行数量就满足要求，那就不需要padding了
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class get_begin_rows_of_BMW_after_nnz_blocking_in_row_direction : public basic_data_transform_step
{
public:
    get_begin_rows_of_BMW_after_nnz_blocking_in_row_direction(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int nnz_per_interval);

    // 执行
    void run(bool check = true);

    int get_nnz_per_interval()
    {
        return nnz_per_interval;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int nnz_per_interval = 0;

    // 需要动态决定，如果一开始的行数量就满足要求，那就不需要padding了
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class get_begin_rows_of_BMTB_after_nnz_blocking_in_row_direction : public basic_data_transform_step
{
public:
    get_begin_rows_of_BMTB_after_nnz_blocking_in_row_direction(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int nnz_per_interval);

    // 执行
    void run(bool check = true);

    int get_nnz_per_interval()
    {
        return nnz_per_interval;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int nnz_per_interval = 0;

    // 需要动态决定，如果一开始的行数量就满足要求，那就不需要padding了
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class get_begin_nzs_of_BMTB_after_nnz_blocking_in_row_direction : public basic_data_transform_step
{
public:
    get_begin_nzs_of_BMTB_after_nnz_blocking_in_row_direction(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int nnz_per_interval);

    // 执行
    void run(bool check = true);

    int get_nnz_per_interval()
    {
        return nnz_per_interval;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int nnz_per_interval = 0;

    // 需要动态决定，如果一开始的行数量就满足要求，那就不需要padding了
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class get_begin_rows_of_BMT_after_fixed_blocking_in_nnz_direction : public basic_data_transform_step
{
public:
    get_begin_rows_of_BMT_after_fixed_blocking_in_nnz_direction(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int nnz_per_BMT);

    // 执行
    void run(bool check = true);

    int get_nnz_per_BMT()
    {
        return nnz_per_BMT;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int nnz_per_BMT = 0;

    // 需要动态决定，如果一开始的行数量就满足要求，那就不需要padding了
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class get_begin_nzs_of_BMT_after_fixed_blocking_in_nnz_direction : public basic_data_transform_step
{
public:
    get_begin_nzs_of_BMT_after_fixed_blocking_in_nnz_direction(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int nnz_per_BMT);

    // 执行
    void run(bool check = true);

    int get_nnz_per_BMT()
    {
        return nnz_per_BMT;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int nnz_per_BMT = 0;

    // 需要动态决定，如果一开始的行数量就满足要求，那就不需要padding了
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class get_begin_rows_of_BMW_after_fixed_blocking_in_nnz_direction : public basic_data_transform_step
{
public:
    get_begin_rows_of_BMW_after_fixed_blocking_in_nnz_direction(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int nnz_per_BMW);

    // 执行
    void run(bool check = true);

    int get_nnz_per_BMW()
    {
        return nnz_per_BMW;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int nnz_per_BMW = 0;

    // 需要动态决定，如果一开始的行数量就满足要求，那就不需要padding了
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class get_begin_nzs_of_BMW_after_fixed_blocking_in_nnz_direction : public basic_data_transform_step
{
public:
    get_begin_nzs_of_BMW_after_fixed_blocking_in_nnz_direction(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int nnz_per_BMW);

    // 执行
    void run(bool check = true);

    int get_nnz_per_BMW()
    {
        return nnz_per_BMW;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int nnz_per_BMW = 0;

    // 需要动态决定，如果一开始的行数量就满足要求，那就不需要padding了
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class get_begin_rows_of_BMTB_after_fixed_blocking_in_nnz_direction : public basic_data_transform_step
{
public:
    get_begin_rows_of_BMTB_after_fixed_blocking_in_nnz_direction(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int nnz_per_BMTB);

    // 执行
    void run(bool check = true);

    int get_nnz_per_BMTB()
    {
        return nnz_per_BMTB;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int nnz_per_BMTB = 0;

    // 需要动态决定，如果一开始的行数量就满足要求，那就不需要padding了
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class get_begin_nzs_of_BMTB_after_fixed_blocking_in_nnz_direction : public basic_data_transform_step
{
public:
    get_begin_nzs_of_BMTB_after_fixed_blocking_in_nnz_direction(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int nnz_per_BMTB);

    // 执行
    void run(bool check = true);

    int get_nnz_per_BMTB()
    {
        return nnz_per_BMTB;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int nnz_per_BMTB = 0;

    // 需要动态决定，如果一开始的行数量就满足要求，那就不需要padding了
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class get_begin_nzs_of_BMT_after_fixed_blocking_in_nnz_direction_relative_to_BMTB : public basic_data_transform_step
{
public:
    get_begin_nzs_of_BMT_after_fixed_blocking_in_nnz_direction_relative_to_BMTB(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int nnz_per_BMT);

    // 执行
    void run(bool check = true);

    int get_nnz_per_BMT()
    {
        return nnz_per_BMT;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int nnz_per_BMT = 0;

    // 需要动态决定，如果一开始的行数量就满足要求，那就不需要padding了
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class get_begin_rows_of_BMT_after_fixed_blocking_in_nnz_direction_relative_to_BMTB : public basic_data_transform_step
{
public:
    get_begin_rows_of_BMT_after_fixed_blocking_in_nnz_direction_relative_to_BMTB(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int nnz_per_BMT);

    // 执行
    void run(bool check = true);

    int get_nnz_per_BMT()
    {
        return nnz_per_BMT;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int nnz_per_BMT = 0;

    // 需要动态决定，如果一开始的行数量就满足要求，那就不需要padding了
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class get_begin_rows_of_BMT_after_fixed_blocking_in_nnz_direction_relative_to_BMW : public basic_data_transform_step
{
public:
    get_begin_rows_of_BMT_after_fixed_blocking_in_nnz_direction_relative_to_BMW(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int nnz_per_BMT);

    // 执行
    void run(bool check = true);

    int get_nnz_per_BMT()
    {
        return nnz_per_BMT;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int nnz_per_BMT = 0;

    // 需要动态决定，如果一开始的行数量就满足要求，那就不需要padding了
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class get_begin_nzs_of_BMT_after_fixed_blocking_in_nnz_direction_relative_to_BMW : public basic_data_transform_step
{
public:
    get_begin_nzs_of_BMT_after_fixed_blocking_in_nnz_direction_relative_to_BMW(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int nnz_per_BMT);

    // 执行
    void run(bool check = true);

    int get_nnz_per_BMT()
    {
        return nnz_per_BMT;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int nnz_per_BMT = 0;

    // 需要动态决定，如果一开始的行数量就满足要求，那就不需要padding了
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class get_begin_nzs_of_BMW_after_fixed_blocking_in_nnz_direction_relative_to_BMTB : public basic_data_transform_step
{
public:
    get_begin_nzs_of_BMW_after_fixed_blocking_in_nnz_direction_relative_to_BMTB(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int nnz_per_BMW);

    // 执行
    void run(bool check = true);

    int get_nnz_per_BMW()
    {
        return nnz_per_BMW;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int nnz_per_BMW = 0;

    // 需要动态决定，如果一开始的行数量就满足要求，那就不需要padding了
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class get_begin_rows_of_BMW_after_fixed_blocking_in_nnz_direction_relative_to_BMTB : public basic_data_transform_step
{
public:
    get_begin_rows_of_BMW_after_fixed_blocking_in_nnz_direction_relative_to_BMTB(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int nnz_per_BMW);

    // 执行
    void run(bool check = true);

    int get_nnz_per_BMW()
    {
        return nnz_per_BMW;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int nnz_per_BMW = 0;

    // 需要动态决定，如果一开始的行数量就满足要求，那就不需要padding了
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class modify_col_indices_by_nnz_pad : public basic_data_transform_step
{
public:
    modify_col_indices_by_nnz_pad(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int nnz_target);

    void run(bool check = true);

    int get_nnz_target()
    {
        return nnz_target;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    string convert_to_string();

    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

private:
    int nnz_target = 0;
    int target_matrix_id = -1;

    // 如果一开始就满足了padding之后的要求，那就不需要是
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class modify_row_indices_by_nnz_pad : public basic_data_transform_step
{
public:
    modify_row_indices_by_nnz_pad(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int nnz_target);

    void run(bool check = true);

    int get_nnz_target()
    {
        return nnz_target;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    string convert_to_string();

    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

private:
    int nnz_target = 0;
    int target_matrix_id = -1;

    // 如果一开始就满足了padding之后的要求，那就不需要是
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class modify_vals_by_nnz_pad : public basic_data_transform_step
{
public:
    modify_vals_by_nnz_pad(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int nnz_target);

    void run(bool check = true);

    int get_nnz_target()
    {
        return nnz_target;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    string convert_to_string();

    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

private:
    int nnz_target = 0;
    int target_matrix_id = -1;

    // 如果一开始就满足了padding之后的要求，那就不需要是
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class thread_bit_map : public basic_data_transform_step
{
public:
    thread_bit_map(shared_ptr<meta_data_set> meta_data_set_ptr, bool parent_flag, int parent_size, int target_matrix_id);

    // 执行
    void run(bool check = true);

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    bool parent_flag = false;
    int parent_size = 0;

    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class parent_bit_map_of_thread : public basic_data_transform_step
{
public:
    parent_bit_map_of_thread(shared_ptr<meta_data_set> meta_data_set_ptr, POS_TYPE pos, int target_matrix_id);

    // 执行
    void run(bool check = true);

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    POS_TYPE pos = NONE_META;
    // 需要动态决定，如果一开始的行数量就满足要求，那就不需要padding了
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class segment_empty_row_indices : public basic_data_transform_step
{
public:
    segment_empty_row_indices(shared_ptr<meta_data_set> meta_data_set_ptr, POS_TYPE pos, int target_matrix_id);

    // 执行
    void run(bool check = true);

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    POS_TYPE pos = NONE_META;

    // 需要动态决定，如果一开始的行数量就满足要求，那就不需要padding了
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class segment_empty_flag : public basic_data_transform_step
{
public:
    segment_empty_flag(shared_ptr<meta_data_set> meta_data_set_ptr, POS_TYPE pos, unsigned int size, int target_matrix_id);

    // 执行
    void run(bool check = true);

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    POS_TYPE pos = NONE_META;
    unsigned int size;

    // 需要动态决定，如果一开始的行数量就满足要求，那就不需要padding了
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class segment_ptr : public basic_data_transform_step
{
public:
    segment_ptr(shared_ptr<meta_data_set> meta_data_set_ptr, POS_TYPE pos, int target_matrix_id);

    // 执行
    void run(bool check = true);

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    POS_TYPE pos = NONE_META;

    // 需要动态决定，如果一开始的行数量就满足要求，那就不需要padding了
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};


class segment_offset : public basic_data_transform_step
{
public:
    segment_offset(shared_ptr<meta_data_set> meta_data_set_ptr, bool parent_flag, int size, int target_matrix_id);

    // 执行
    void run(bool check = true);

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int size = 0;
    bool parent_flag = false;

    // 需要动态决定，如果一开始的行数量就满足要求，那就不需要padding了
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};



class get_begin_rows_after_merge_thread : public basic_data_transform_step
{
public:
    get_begin_rows_after_merge_thread(shared_ptr<meta_data_set> meta_data_set_ptr, POS_TYPE pos, int merge_num, int target_matrix_id);

    // 执行
    void run(bool check = true);

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    POS_TYPE pos = NONE_META;
    int merge_num;

    // 需要动态决定，如果一开始的行数量就满足要求，那就不需要padding了
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class get_begin_nzs_after_merge_thread : public basic_data_transform_step
{
public:
    get_begin_nzs_after_merge_thread(shared_ptr<meta_data_set> meta_data_set_ptr, POS_TYPE pos, int merge_num, int target_matrix_id);

    // 执行
    void run(bool check = true);

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    POS_TYPE pos = NONE_META;
    int merge_num;

    // 需要动态决定，如果一开始的行数量就满足要求，那就不需要padding了
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class get_begin_rows_relative_to_parent_after_merge_thread : public basic_data_transform_step
{
public:
    get_begin_rows_relative_to_parent_after_merge_thread(shared_ptr<meta_data_set> meta_data_set_ptr, POS_TYPE pos, int merge_num, int target_matrix_id);

    // 执行
    void run(bool check = true);

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    POS_TYPE pos = NONE_META;
    int merge_num;

    // 需要动态决定，如果一开始的行数量就满足要求，那就不需要padding了
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class get_begin_nzs_relative_to_parent_after_merge_thread : public basic_data_transform_step
{
public:
    get_begin_nzs_relative_to_parent_after_merge_thread(shared_ptr<meta_data_set> meta_data_set_ptr, POS_TYPE pos, int merge_num, int target_matrix_id);

    // 执行
    void run(bool check = true);

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    POS_TYPE pos = NONE_META;
    int merge_num;

    // 需要动态决定，如果一开始的行数量就满足要求，那就不需要padding了
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

class get_begin_BMTs_after_merge_thread : public basic_data_transform_step
{
public:
    get_begin_BMTs_after_merge_thread(shared_ptr<meta_data_set> meta_data_set_ptr, POS_TYPE pos, int merge_num, int target_matrix_id);

    // 执行
    void run(bool check = true);

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    POS_TYPE pos = NONE_META;
    int merge_num;

    // 需要动态决定，如果一开始的行数量就满足要求，那就不需要padding了
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};



class get_begin_rows_of_level_after_merge_path : public basic_data_transform_step
{
public:
    get_begin_rows_of_level_after_merge_path(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, POS_TYPE pos, int work_size);

    // 执行
    void run(bool check = true);

    int get_work_size()
    {
        return work_size;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int work_size = 0;
    POS_TYPE pos = NONE_META;

    // 需要动态决定，如果一开始的行数量就满足要求，那就不需要padding了
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};


class get_begin_nzs_of_level_after_merge_path : public basic_data_transform_step
{
public:
    get_begin_nzs_of_level_after_merge_path(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, POS_TYPE pos, int work_size);

    // 执行
    void run(bool check = true);

    int get_work_size()
    {
        return work_size;
    }

    int get_target_matrix_id()
    {
        return target_matrix_id;
    }

    // 获取输入数据的记录
    vector<shared_ptr<data_item_record>> get_source_data_item_ptr_in_data_transform_step();

    string convert_to_string();

private:
    int target_matrix_id = -1;
    int work_size = 0;
    POS_TYPE pos = NONE_META;

    // 需要动态决定，如果一开始的行数量就满足要求，那就不需要padding了
    vector<shared_ptr<data_item_record>> source_data_item_ptr_vec;
    vector<shared_ptr<data_item_record>> dest_data_item_ptr_vec;

    vector<shared_ptr<data_item_record>> get_dest_data_item_ptr_in_data_transform_step_without_check();
};

// BMT的列切块，分为padding和不padding两种方式，并且根据不同类型的父块给出不同padding之后的偏移量。按照相等的大小切分
// class get_begin_rows_of_BMT_after_fixed_blocking_in_col_direction : public basic_data_transform_step
// {
// public:
//     get_begin_rows_of_BMT_after_fixed_blocking_in_col_direction(shared_ptr<meta_data_set> meta_data_set_ptr, int target_matrix_id, int fixed_col_block_size, bool is_padding);

//     //

// private:

// };

// 获得相对于父块的行索引

// BMT级别的padding，将同一个父块中的BMT的大小padding到一致，相对于全局的，BMTB的padding会导致只存储每个父块中BMT的大小

// BMT级别的padding，相对于上一层BMW的

// BMT级别的padding，相对于BMTB的

#endif