#include "../kernel_generator.h"

math_expr_token::math_expr_token(string math_expression_str)
:basic_token(true, MATH_EXPR_TOKEN_TYPE)
{
    this->math_expression_str = math_expression_str;
}

math_expr_token::math_expr_token(string math_expression_str, shared_ptr<arr_access_token> arr_acc_expr, string op)
:basic_token(false, MATH_EXPR_TOKEN_TYPE)
{
    this->math_expression_str = math_expression_str;
    this->token_of_child_map["access_token"] = arr_acc_expr;
    this->op = op;
}

string math_expr_token::run()
{
    string return_str = this->math_expression_str;
    string acc_token_string;
    if(this->child_is_exist("access_token") == true)
    {
        acc_token_string = this->get_token_of_child("access_token")->run(); 
        int count = acc_token_string.find("=") + 1;
        acc_token_string.erase(0, count);
        acc_token_string.pop_back();
        return_str += this->op + acc_token_string;
    }

    assert(this->static_check() == true);

    return return_str;
}

bool math_expr_token::static_check()
{
    // 遍历字符串，做一些简单的检查
    for (int i = 0; i < this->math_expression_str.size(); i++)
    {
        if (this->math_expression_str[i] == ',' || this->math_expression_str[i] == ';')
        {
            return false;
        }

        if (this->math_expression_str[i] == '\n')
        {
            return false;
        }

    }

    return true;
}