#ifndef IO_OF_REDUCTION_HPP
#define IO_OF_REDUCTION_HPP

#include "kernel_generator.h"

// 输入和输出在同一个寄存器中，通常是THREAD级别的输出，WARP级别的输入，WARP级别的输出，TBLOCK级别的输出
// 而这些输出的变量在reduction token中都是存在的，只是变量名需要按照IO中的声明来
class one_register_result_IO_of_reduction : public basic_IO_of_reduction
{
public:
    one_register_result_IO_of_reduction(POS_TYPE pos, int count);
    unsigned int get_count();

    // IO内部的变量必须只是一个写死的字符串，不能和任何参数有关系，
    shared_ptr<var_name_token> var_name_token_of_IO_register();
private:
    int count = 0;
};


class two_register_result_IO_of_reduction : public basic_IO_of_reduction
{
public:
    two_register_result_IO_of_reduction(POS_TYPE pos, int count);
    unsigned int get_count();

    // IO内部的变量必须只是一个写死的字符串，不能和任何参数有关系，
    vector<shared_ptr<var_name_token>> var_names_token_of_IO_register();
private:
    int count = 0;
};


#endif