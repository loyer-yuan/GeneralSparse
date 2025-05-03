#include "../kernel_generator.h"

for_token::for_token(shared_ptr<data_type_token> loop_var_name_type, shared_ptr<var_name_token> loop_var_name,
                     shared_ptr<var_name_token> begin_loop_var_name, shared_ptr<var_name_token> end_loop_var_name,
                     shared_ptr<var_name_token> step_loop_var_name, POS_TYPE token_position, shared_ptr<metadata_get_basic_token> metadata_get_code,
                     shared_ptr<for_basic_token> inner_loop, shared_ptr<reduction_basic_token> reduction_code, shared_ptr<basic_glue_code> glue_code_block)
    : for_basic_token(token_position, metadata_get_code, inner_loop, reduction_code, glue_code_block)
{
    // 当前子类的所有的输入都不能是空的
    assert(loop_var_name_type != NULL && loop_var_name != NULL && begin_loop_var_name != NULL && end_loop_var_name != NULL && step_loop_var_name != NULL);
    // 将token放到map中
    this->token_of_child_map["loop_var_name_type"] = loop_var_name_type;
    this->token_of_child_map["loop_var_name"] = loop_var_name;
    this->token_of_child_map["begin_loop_var_name"] = begin_loop_var_name;
    this->token_of_child_map["end_loop_var_name"] = end_loop_var_name;
    this->token_of_child_map["step_loop_var_name"] = step_loop_var_name;

    this->token_type = FOR_TOKEN_TYPE;
}


for_token::for_token(shared_ptr<data_type_token> loop_var_name_type, shared_ptr<var_name_token> loop_var_name,
                     shared_ptr<basic_token> begin_loop_var_name, shared_ptr<basic_token> end_loop_var_name,
                     shared_ptr<var_name_token> step_loop_var_name, POS_TYPE token_position, shared_ptr<metadata_get_basic_token> metadata_get_code,
                     shared_ptr<for_basic_token> inner_loop, shared_ptr<reduction_basic_token> reduction_code, shared_ptr<basic_glue_code> glue_code_block)
    : for_basic_token(token_position, metadata_get_code, inner_loop, reduction_code, glue_code_block)
{
    // 当前子类的所有的输入都不能是空的
    assert(loop_var_name_type != NULL && loop_var_name != NULL && begin_loop_var_name != NULL && end_loop_var_name != NULL && step_loop_var_name != NULL);
    // 将token放到map中
    this->token_of_child_map["loop_var_name_type"] = loop_var_name_type;
    this->token_of_child_map["loop_var_name"] = loop_var_name;
    this->token_of_child_map["begin_loop_var_name"] = begin_loop_var_name;
    this->token_of_child_map["end_loop_var_name"] = end_loop_var_name;
    this->token_of_child_map["step_loop_var_name"] = step_loop_var_name;

    this->token_type = FOR_TOKEN_TYPE;
}

string for_token::for_header_run()
{
    // 静态检查一定能过
    assert(this->static_check() == true);

    string return_str = "for (";

    // 首先拼接遍历变量的数据类型
    return_str = return_str + this->token_of_child_map["loop_var_name_type"]->run();
    // 拼接初始化的变量
    return_str = return_str + " " + this->token_of_child_map["loop_var_name"]->run() + " = ";
    // 拼接遍历的起始位置
    return_str = return_str + this->token_of_child_map["begin_loop_var_name"]->run() + "; ";

    // 结束位置的判断
    return_str = return_str + this->token_of_child_map["loop_var_name"]->run() + " < ";
    return_str = return_str + this->token_of_child_map["end_loop_var_name"]->run() + "; ";
    
    // 遍历变量的自增
    return_str = return_str + this->token_of_child_map["loop_var_name"]->run() + " = " + this->token_of_child_map["loop_var_name"]->run() + " + " + this->token_of_child_map["step_loop_var_name"]->run();

    return_str = return_str + ")";

    return return_str;
}

bool for_token::static_check()
{
    // 首先执行父类的检查
    if (for_basic_token::static_check() == false)
    {
        return false;
    }

    // 子类的检查，首先循环的数据类型不能是浮点及bool类型，并且不能是指针
    shared_ptr<data_type_token> loop_var_name_type = dynamic_pointer_cast<data_type_token>(this->token_of_child_map["loop_var_name_type"]);
    assert(loop_var_name_type->get_token_type() == DATA_TYPE_TOKEN_TYPE);

    // 直接对数据类型执行检查
    if (loop_var_name_type->static_check() == false)
    {
        cout << "for_token::static_check(): invaild loop_var_name_type" << endl;
        return false;
    }
    
    // 首先不能是指针
    if (loop_var_name_type->get_is_pointer() == true)
    {
        cout << "for_token::static_check(): loop_var_name_type should not be a pointer" << endl;
        return false;
    }

    // 不能是不能迭代的数据类型
    if (loop_var_name_type->get_data_type() == BOOL || loop_var_name_type->get_data_type() == FLOAT || loop_var_name_type->get_data_type() == DOUBLE)
    {
        cout << "for_token::static_check(): loop_var_name_type is not iterable. loop_var_name_type->get_data_type():" << convert_data_type_to_string(loop_var_name_type->get_data_type()) << endl;
        return false;
    }

    shared_ptr<var_name_token> loop_var_name = dynamic_pointer_cast<var_name_token>(this->token_of_child_map["loop_var_name"]);
    assert(loop_var_name->get_token_type() == VAR_NAME_TOKEN_TYPE);

    // 检查迭代变量
    if (loop_var_name->static_check() == false)
    {
        cout << "for_token::static_check(): invaild loop_var_name" << endl;
        return false;
    }

    // 检查是不是register类型
    if (loop_var_name->get_var_type() != REGISTER_VAR_TYPE)
    {
        cout << "for_token::static_check(): loop_var_name is not REGISTER_VAR_TYPE" << endl;
        return false;
    }

    return true;
}