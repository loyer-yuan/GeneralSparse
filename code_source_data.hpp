#ifndef CODE_SOUCE_DATA_H
#define CODE_SOUCE_DATA_H

#include "struct.hpp"
#include "op_manager.hpp"
#include <vector>
#include <memory>
#include <iostream>

// 通用的、支持任何内容的数组
class universal_array
{
public:
    // 传入指针、数据类型、长度。使用指针保证可以使用shared_ptr、vector来传入内容，vector一般是栈区对象，shared_ptr是堆区的。
    // 只能值传递
    universal_array(void *input_arr_ptr, unsigned long len, data_type type);

    // 析构
    ~universal_array();

    unsigned long get_len()
    {
        return this->len;
    }

    data_type get_data_type()
    {
        return this->type;
    }

    data_type get_compress_data_type();

    // 从特定位置读整型数据
    unsigned long read_integer_from_arr(unsigned long read_index);
    // 从特定位置读浮点类型
    double read_float_from_arr(unsigned long read_index);

    // 压缩数据类型
    void compress_data_type();

    // 压缩精度，仅仅针对浮点类型，会改变语义的操作
    void compress_float_precise();

    // 数据成员是不是合法
    bool check();

    // 将数据输出到任意文件中
    void output_2_file(string file_name);

    // 获得最大的整型
    unsigned long get_max_integer();

    // 获得最小的整型
    unsigned long get_min_integer();

    // 获得最大浮点
    double get_max_float();

    // 获得最小浮点
    double get_min_float();

    void copy_to(void * dst_pos, int src_pos, unsigned int source_size, data_type type);
    
    void * get_arr_ptr(data_type type);

private:
    void *arr_ptr = NULL;
    unsigned long len = 0;
    data_type type;

    // 暂时关闭写的功能
    // 向特定位置写数据，分别是整数和浮点数
    void write_integer_to_arr(unsigned long input_val, unsigned long input_index);
    // 向特定位置写数据，处理浮点类型
    void write_float_to_arr(double input_val, unsigned long input_index);
};

// 模板类和模板函数的声明和定义要写在同一个文件。不然编译会出现“未定义的引用”。
// 主要原因是模板不是真正的代码，需要在编译过程中转化为对应的类的声明和实现。如果把模板的声明和实现分别放到h和cc中，在其他文件
// 模板中的数据类型可以被确定的时候，可能cc文件还没被编译器看到，那么编译器就没有办法去生成其对应的真实实现。编译器不会对所有代码进行通篇处理
// 而是一个文件一个文件进行处理的。那么在模板中对应的空缺被编译器识别出来，模板可以转化为实际代码的时候，对应的模板实现应该也要被看到，所以只能全在头文件里面。
// cc文件是分开编译的。h文件本质上会和cc文件合并成一个文件。
// https://blog.csdn.net/cllcsy/article/details/50485324
// template <typename T>
// shared_ptr<universal_array> create_uni_arr_from_vec(vector<T> input_vec, bool need_compress = false)
// {
//     assert(input_vec.size() > 0);
//     // 只支持无符号整型、bool和浮点
//     assert(typeid(T) == typeid(bool) || typeid(T) == typeid(unsigned char) || typeid(T) == typeid(unsigned short) ||
//            typeid(T) == typeid(unsigned int) || typeid(T) == typeid(unsigned long) || typeid(T) == typeid(float) || typeid(T) == typeid(double));

//     // 整型和浮点分开处理，只有整型可以压缩
//     bool is_integer = (typeid(T) == typeid(bool) || typeid(T) == typeid(unsigned char) || typeid(T) == typeid(unsigned short) ||
//                        typeid(T) == typeid(unsigned int) || typeid(T) == typeid(unsigned long));

//     bool is_float = (typeid(T) == typeid(float) || typeid(T) == typeid(double));

//     assert((is_integer && is_float) == false);
//     assert((is_integer || is_float) == true);

//     if (is_float == true)
//     {
//         assert(need_compress == false);
//         // 创造一个指针，然后分别处理
//         data_type val_type = get_data_type_from_type_info(typeid(T));
//         // 创造指针
//         void *val_arr = malloc_arr(input_vec.size(), val_type);
//         // 将指针的内容拷贝进来
//         for (unsigned long i = 0; i < input_vec.size(); i++)
//         {
//             // 用新的指针存储数据
//             write_double_to_array_with_data_type(val_arr, val_type, i, input_vec[i]);
//         }

//         // 初始化
//         shared_ptr<universal_array> return_ptr(new universal_array(val_arr, input_vec.size(), val_type, false));

//         // 返回对应的指针
//         return return_ptr;
//     }
//     else if (is_integer == true)
//     {
//         // 创造一个指针，然后分别处理
//         data_type val_type = get_data_type_from_type_info(typeid(T));

//         // 查看数据类型的压缩
//         if (need_compress == true)
//         {
//             assert(typeid(T) != typeid(bool));

//             // 这里是遇到整型的情况，需要处理可能的小数据类型问题
//             T max_val = *max_element(input_vec.begin(), input_vec.end());
//             T min_val = *min_element(input_vec.begin(), input_vec.end());

//             // 暂时只支持无符号
//             assert(min_val >= 0);

//             val_type = find_most_suitable_data_type(max_val);
//         }

//         // 创造指针
//         void *val_arr = malloc_arr(input_vec.size(), val_type);
//         // 将指针的内容拷贝进来
//         for (unsigned long i = 0; i < input_vec.size(); i++)
//         {
//             // 用新的指针存储数据
//             write_to_array_with_data_type(val_arr, val_type, i, input_vec[i]);
//         }

//         // 初始化
//         shared_ptr<universal_array> return_ptr(new universal_array(val_arr, input_vec.size(), val_type, false));

//         // 返回对应的指针
//         return return_ptr;
//     }
//     else
//     {
//         assert(false);
//     }
// }


// // 查看常量类型，需要给出等号右边的值怎么处理
// class constant_data_source_item : public data_source_item
// {
// public:
//     // 构造函数，需要足以构造父类
//     // constant_data_source_item(string name, POS_TYPE position_type) : data_source_item(name, position_type)
//     // {
//     //     // 空的构造函数
//     // }

//     // 构造函数，传入是右值是不是CUDA自带的东西，并且声明数据类型
//     constant_data_source_item(string name, POS_TYPE position_type, bool is_right_value_from_CUDA, data_type data_type_of_constant_data_source)
//         : data_source_item(name, position_type), 
//         is_right_value_from_CUDA(is_right_value_from_CUDA),
//         data_type_of_constant_data_source(data_type_of_constant_data_source)
//     {
//         // 空的构造函数
//     }

//     // 设置右值的名称
//     void set_right_val_name(string right_value_name)
//     {
//         this->right_value_name = right_value_name;
//     }
    
//     virtual string get_define_code();

//     virtual string get_assign_code();

// protected:
//     // 右值
//     string right_value_name = "";
//     // 右值是不是来源于CUDA自带的东西，用来执行之后的变量依赖检查
//     bool is_right_value_from_CUDA = false;
//     // 数据类型
//     data_type data_type_of_constant_data_source = NONE_DATA_TYPE;
// };



#endif