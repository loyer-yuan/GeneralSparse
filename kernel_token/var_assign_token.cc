#include "../kernel_generator.h"

var_assign_token::var_assign_token(shared_ptr<var_name_token> left_operand, shared_ptr<var_name_token> right_operand)
    : basic_token(false, VAR_ASSIGN_TOKEN_TYPE)
{
    assert(left_operand != NULL);
    assert(right_operand != NULL);

    this->token_of_child_map["left_operand"] = left_operand;
    this->token_of_child_map["right_operand"] = right_operand;
}

var_assign_token::var_assign_token(shared_ptr<var_name_token> left_operand, shared_ptr<math_expr_token> right_operand)
    : basic_token(false, VAR_ASSIGN_TOKEN_TYPE)
{
    assert(left_operand != NULL);
    assert(right_operand != NULL);

    this->token_of_child_map["left_operand"] = left_operand;
    this->token_of_child_map["right_operand"] = right_operand;
}

// 实际上执行复制，先左后右，最后有分号
string var_assign_token::run()
{
    assert(this->child_is_exist("left_operand"));
    assert(this->child_is_exist("right_operand"));
    assert(this->static_check());

    string return_str = this->get_token_of_child("left_operand")->run() + " = " + this->get_token_of_child("right_operand")->run() + ";";
    
    return return_str;
}

// 执行静态检查
bool var_assign_token::static_check()
{
    // 首先看左边和右边
    assert(this->child_is_exist("left_operand"));
    
    // 将左边的元素取出来
    shared_ptr<var_name_token> left_operand = dynamic_pointer_cast<var_name_token>(this->get_token_of_child("left_operand"));
    assert(left_operand->get_token_type() == VAR_NAME_TOKEN_TYPE);

    // 递归静态检查
    if (left_operand->static_check() == false)
    {
        cout << "var_assign_token::static_check(): invalid left_operand" << endl;
        return false;
    }

    // 左操作数必须是寄存器的
    if (left_operand->get_var_type() != REGISTER_VAR_TYPE)
    {
        cout << "var_assign_token::static_check(): left_operand is not REGISTER_VAR_TYPE" << endl;
        return false;
    }

    // 右操作数
    assert(this->child_is_exist("right_operand"));

    shared_ptr<basic_token> basic_right_operand = this->get_token_of_child("right_operand");
    assert(basic_right_operand->get_token_type() == VAR_NAME_TOKEN_TYPE || basic_right_operand->get_token_type() == MATH_EXPR_TOKEN_TYPE);

    // 递归检查
    if (basic_right_operand->static_check() == false)
    {
        cout << "var_assign_token::static_check(): invalid right_operand" << endl;
        return false;
    }

    // 如果是变量，必须是寄存器和常量
    if (basic_right_operand->get_token_type() == VAR_NAME_TOKEN_TYPE)
    {
        // 强制类型转换
        shared_ptr<var_name_token> right_operand = dynamic_pointer_cast<var_name_token>(basic_right_operand);
        
        // 必须是寄存器和常量
        if (right_operand->get_var_type() != REGISTER_VAR_TYPE && right_operand->get_var_type() != CONSTANT_VAR_TYPE)
        {
            cout << "var_assign_token::static_check(): right_operand is not REGISTER or CONSTANT" << endl;
            return false;
        }
    }

    return true;
}
