#include "../kernel_generator.h"

data_type_token::data_type_token(data_type type, bool is_pointer)
:basic_token(true, DATA_TYPE_TOKEN_TYPE)
{
    this->type = type;
    this->is_pointer = is_pointer;
}

// 执行
string data_type_token::run()
{
    // 将对应的数据类型转化为具体的代码
    assert(this->static_check() == true);

    string return_str = "";
    
    return_str = return_str + code_of_data_type(this->type);

    // 查看是不是指针
    if (this->is_pointer == true)
    {
        return_str = return_str + "*";
    }

    // 返回
    return return_str;
}

bool data_type_token::static_check()
{
    // 必然通过，没有用户需要自己填写的字符串
    return true;
}