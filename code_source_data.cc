#include "code_source_data.hpp"
#include <typeinfo>
#include <assert.h>
#include <algorithm>
#include "config.hpp"

using namespace std;

// 关于通用数组类型的实现。
// universal_array::universal_array(vector<T> input_vec, data_type suggest_type, bool need_compress)
// {
//     assert(input_vec.size() > 0);

//     // 首先查看取值范围是不是和建议的数据类型吻合
//     if (suggest_type == BOOL)
//     {
//         assert(typeid(T).name() == typeid(bool).name());
//     }
//     else if (suggest_type == DOUBLE)
//     {
//         // 这里只是精度问题
//         assert(typeid(T).name() == typeid(double).name());
//     }
//     else if (suggest_type == FLOAT)
//     {
//         assert(typeid(T).name() == typeid(float).name());
//     }
//     else
//     {
//         // 这里是遇到整型的情况，需要处理可能的小数据类型问题
//         T max_val = *max_element(input_vec.begin(), input_vec.end());
//         T min_val = *min_element(input_vec.begin(), input_vec.end());

//         // 暂时只支持无符号
//         assert(min_val >= 0);

//         // 对于整数来说，如果没有设定自动压缩，就要看手动设定的数据类型是不是满足要求
//         // 获取数组的最大值和最小值
//         if (need_compress == false)
//         {
//             assert(max_val <= get_max_of_a_integer_data_type(suggest_type));
//             assert(min_val >= get_min_of_a_integer_data_type(suggest_type));
//         }
//         else
//         {
//             // 如果自带压缩，就直接找压缩的方法
//             if (min_val >= 0)
//             {
//                 suggest_type = find_most_suitable_data_type(max_val);
//             }
//             else
//             {
//                 suggest_type = find_most_suitable_data_type(max_val, min_val);
//             }
//         }
//     }

//     // 到这里只支持无符号、bool和浮点
//     assert(suggest_type == FLOAT || suggest_type == DOUBLE || suggest_type == UNSIGNED_CHAR || suggest_type == UNSIGNED_SHORT ||
//         suggest_type == UNSIGNED_INT || suggest_type == UNSIGNED_LONG || suggest_type == BOOL);

//     // 现在suggest_type中存着数据类型
//     this->type = suggest_type;
//     this->len = input_vec.size();

//     this->arr_ptr = malloc_arr(this->len, this->type);

//     assert(this->arr_ptr != NULL);

//     // 遍历所有的将向量中所有的非零元拷贝到新的数组中
//     for (unsigned long i = 0; i < input_vec.size(); i++)
//     {
//         // 浮点类型
//         if (this->type == FLOAT || this->type == DOUBLE)
//         {
//             assert((typeid(T).name() == typeid(float).name()) || (typeid(T).name() == typeid(double).name()));
//             write_double_to_array_with_data_type(this->arr_ptr, this->type, i, input_vec[i]);
//         }
//         else
//         {
//             write_to_array_with_data_type(this->arr_ptr, this->type, i, input_vec[i]);
//         }
//     }

//     cout << "universal_array::universal_array: the final data type is " << convert_data_type_to_string(this->type) << endl;
// }

// 利用引用的方式来初始化，不拷贝
universal_array::universal_array(void *input_arr_ptr, unsigned long len, data_type type)
{
    assert(input_arr_ptr != NULL);
    assert(len > 0);
    // 支持浮点类型，无符号整型，bool类型
    assert(type == UNSIGNED_CHAR || type == UNSIGNED_SHORT || type == UNSIGNED_INT || type == UNSIGNED_LONG || type == FLOAT || type == DOUBLE);

    // if (copy == false)
    // {
    //     // 引用传值
    //     this->arr_ptr = input_arr_ptr;
    //     this->len = len;
    //     this->type = type;
    // }
    // else
    // {
    // 申请新的空间，并且执行拷贝
    this->arr_ptr = val_copy_from_old_arr_with_data_type(input_arr_ptr, len, type);
    this->len = len;
    this->type = type;

    assert(this->arr_ptr != NULL);
    // }

    assert(check());
}

bool universal_array::check()
{
    if (this->arr_ptr == NULL)
    {
        cout << "universal_array::check(): arr_ptr is empty ptr" << endl;
        return false;
    }

    if (this->len <= 0)
    {
        cout << "universal_array::check(): len error, this->len = " << this->len << endl;
        return false;
    }

    if (!(this->type == UNSIGNED_CHAR || this->type == UNSIGNED_SHORT || this->type == UNSIGNED_INT || this->type == UNSIGNED_LONG ||
          this->type == FLOAT || this->type == DOUBLE))
    {
        cout << "universal_array::check(): type error" << endl;
        return false;
    }

    return true;
}

void universal_array::output_2_file(string file_name)
{
    assert(check());
    // 将数组内容输出到任意的文件
    print_arr_to_file_with_data_type(this->arr_ptr, this->get_data_type(), this->get_len(), file_name);
}

universal_array::~universal_array()
{
    assert(check());

    // 析构
    delete_arr_with_data_type(this->arr_ptr, this->type);
}

void universal_array::write_integer_to_arr(unsigned long input_val, unsigned long input_index)
{
    assert(check());
    // 不能是浮点类型
    assert(this->type != FLOAT && this->type != DOUBLE);
    assert(input_index < this->len);

    write_to_array_with_data_type(this->arr_ptr, this->type, input_index, input_val);
}

void universal_array::write_float_to_arr(double input_val, unsigned long input_index)
{
    assert(check());
    // 必须是浮点类型
    assert(this->type == FLOAT || this->type == DOUBLE);
    assert(input_index < this->len);

    write_double_to_array_with_data_type(this->arr_ptr, this->type, input_index, input_val);
}

unsigned long universal_array::read_integer_from_arr(unsigned long read_index)
{
    assert(check());
    // 不能是浮点类型
    assert(this->type != FLOAT && this->type != DOUBLE);

    if (read_index >= this->len)
    {
        cout << "read_index:" << read_index << ",this->len:" << this->len << endl;
        assert(read_index < this->len);
    }

    return read_from_array_with_data_type(this->arr_ptr, this->type, read_index);
}

double universal_array::read_float_from_arr(unsigned long read_index)
{
    assert(check());
    // 必须是浮点类型
    assert(this->type == FLOAT || this->type == DOUBLE);
    assert(read_index < this->len);

    return read_double_from_array_with_data_type(this->arr_ptr, this->type, read_index);
}

unsigned long universal_array::get_max_integer()
{
    assert(check());
    // 不能是浮点类型
    assert(this->type != FLOAT && this->type != DOUBLE);

    unsigned long max = read_integer_from_arr(0);

    // 遍历所有的元素
    for (unsigned long i = 0; i < this->get_len(); i++)
    {
        unsigned long cur_num = read_integer_from_arr(i);

        if (cur_num > max)
        {
            max = cur_num;
        }
    }

    return max;
}

unsigned long universal_array::get_min_integer()
{
    assert(check());
    // 不能是浮点类型
    assert(this->type != FLOAT && this->type != DOUBLE);

    unsigned long min = read_integer_from_arr(0);

    // 遍历所有的元素
    for (unsigned long i = 0; i < this->get_len(); i++)
    {
        unsigned long cur_num = read_integer_from_arr(i);

        if (cur_num < min)
        {
            min = cur_num;
        }
    }

    return min;
}

double universal_array::get_max_float()
{
    assert(check());
    assert(this->type == FLOAT || this->type == DOUBLE);

    double max = read_float_from_arr(0);

    // 遍历所有的元素
    for (unsigned long i = 0; i < this->get_len(); i++)
    {
        double cur_num = read_float_from_arr(i);

        if (cur_num > max)
        {
            max = cur_num;
        }
    }

    return max;
}

double universal_array::get_min_float()
{
    assert(check());
    assert(this->type == FLOAT || this->type == DOUBLE);

    double min = read_float_from_arr(0);

    // 遍历所有元素
    for (unsigned long i = 0; i < this->get_len(); i++)
    {
        double cur_num = read_float_from_arr(i);

        if (cur_num < min)
        {
            min = cur_num;
        }
    }

    return min;
}

// 将已有的数组进行压缩
void universal_array::compress_data_type()
{
    assert(check());
    assert(this->type != FLOAT && this->type != DOUBLE && this->type != BOOL);

    // 找出当前数组的最大值
    unsigned long max_num = 0;

    // 遍历当前所有的元素
    for (unsigned long i = 0; i < this->len; i++)
    {
        unsigned long cur_num = this->read_integer_from_arr(i);

        if (cur_num > max_num)
        {
            max_num = cur_num;
        }
    }

    // 用最大值获取当前需要的数据类型
    data_type new_type = find_most_suitable_data_type(max_num);

    // 老的数组指针
    void *old_arr_ptr = this->arr_ptr;
    data_type old_type = this->type;

    // 新的指针
    this->arr_ptr = malloc_arr(this->len, new_type);
    this->type = new_type;

    // 老的数据拷贝到新的指针
    for (unsigned long i = 0; i < this->len; i++)
    {
        unsigned long old_num = read_from_array_with_data_type(old_arr_ptr, old_type, i);

        // 写到新的数组里面
        write_to_array_with_data_type(this->arr_ptr, this->type, i, old_num);
    }

    // 析构老的指针
    delete_arr_with_data_type(old_arr_ptr, old_type);
}

void universal_array::compress_float_precise()
{
    assert(check());
    assert(this->type == DOUBLE);

    // 旧的、double类型的数组
    void *old_arr_ptr = this->arr_ptr;
    // 新的、float类型的数组
    this->arr_ptr = malloc_arr(this->len, FLOAT);
    this->type = FLOAT;

    // 老的数据拷贝到新的指针
    for (unsigned long i = 0; i < this->len; i++)
    {
        double old_num = read_double_from_array_with_data_type(old_arr_ptr, DOUBLE, i);

        // 写到新的数组里面
        write_double_to_array_with_data_type(this->arr_ptr, this->type, i, old_num);
    }

    // 析构老的指针
    delete_arr_with_data_type(old_arr_ptr, FLOAT);
}

void universal_array::copy_to(void *dst_pos, int src_index, unsigned int source_size, data_type type)
{
    if (type == UNSIGNED_INT)
    {
        assert(type == this->get_data_type());
        memcpy_with_data_type(dst_pos, (void *)((unsigned int *)this->arr_ptr + src_index), source_size, type);
        return;
    }
    else if (type == FLOAT)
    {
        assert(type == this->get_data_type());
        memcpy_with_data_type(dst_pos, (void *)((float *)this->arr_ptr + src_index), source_size, type);
        return;
    }
    assert(false);
}

void *universal_array::get_arr_ptr(data_type type)
{
    if (type == this->type)
    {
        return arr_ptr;
    }
    else
    {
        assert(false);
    }
}

data_type universal_array::get_compress_data_type()
{
    if (this->type == FLOAT)
    {
        return FLOAT;
    }
    else if (this->type == DOUBLE)
    {

        return DOUBLE;
    }

    if(get_config()["DATA_TYPE_COMPRESS"].as_bool() == false)
    {
        return this->type;
    }
    // 找出当前数组的最大值
    unsigned long max_num = 0;

    // 遍历当前所有的元素
    for (unsigned long i = 0; i < this->len; i++)
    {
        unsigned long cur_num = this->read_integer_from_arr(i);

        if (cur_num > max_num)
        {
            max_num = cur_num;
        }
    }

    // 用最大值获取当前需要的数据类型
    data_type new_type = find_most_suitable_data_type(max_num);
    return new_type;
}

// int var_code_position_type::global = 1;
// int var_code_position_type::tblock = 2;
// int var_code_position_type::warp = 3;
// int var_code_position_type::thread = 4;
// int var_code_position_type::col_index = 5;
// int var_code_position_type::val = 6;
// int var_code_position_type::none = 7;

// data_source_item::data_source_item(string name, POS_TYPE position_type)
// {
//     this->name = name;
//     this->position_type = position_type;
// }

// string constant_data_source_item::get_define_code()
// {
//     // 左右值都是存在的，位置和数据类型都没有问题
//     assert(this->type_of_data_source == CONS_DATA_SOURCE);
//     assert(this->data_type_of_constant_data_source != NONE_DATA_TYPE);
//     assert(this->name != "" && this->right_value_name != "");
//     assert(this->position_type != var_code_position_type::none);

//     string return_str = "";

//     return_str = return_str + convert_data_type_to_string(this->data_type_of_constant_data_source);
//     return_str = return_str + " " + this->name + ";";

//     return return_str;
// }

// string constant_data_source_item::get_assign_code()
// {
//     // 左右值都是存在的，位置和数据类型都没有问题
//     assert(this->type_of_data_source == CONS_DATA_SOURCE);
//     assert(this->data_type_of_constant_data_source != NONE_DATA_TYPE);
//     assert(this->name != "" && this->right_value_name != "");
//     assert(this->position_type != var_code_position_type::none);

//     string return_str = "";

//     return_str = return_str + this->name + " = " + this->right_value_name + ";";

//     return return_str;
// }
