#include "../kernel_generator.h"
#include "../data_transform_common.hpp"
#include "../code_generator.hpp"

// 本质上是一个终结符，直接是一个字符串，只有输入输出需要按照要求界定
reduction_basic_token::reduction_basic_token(POS_TYPE token_position, shared_ptr<meta_data_set> meta_data_set_ptr, shared_ptr<code_generator> code_generator_ptr)
: basic_token(true, REDUCTION_BASIC_TOKEN_TYPE)
{
    assert(check_pos_type(token_position) == true);
    assert(token_position != NONE_META);

    if (code_generator_ptr != NULL)
    {
        assert(code_generator_ptr->check() == true);
    }

    if (meta_data_set_ptr != NULL)
    {
        assert(meta_data_set_ptr->check() == true);
    }
    
    this->token_position = token_position;
    this->code_generator_ptr = code_generator_ptr;
    this->meta_data_set_ptr = meta_data_set_ptr;
}

// 添加一行AlphaSparse标准的代码
void reduction_basic_token::add_alpha_code_line(shared_ptr<basic_token> alpha_code_line)
{
    cout << "reduction_basic_token::add_alpha_code_line: this interface is deprecated" << endl;
    assert(false);
    // 首先看最后是不是有分号，代表是不是一个完整的代码
    assert(alpha_code_line != NULL);
    assert(alpha_code_line->get_token_type() != RAW_CUDA_CODE_LINE && alpha_code_line->get_token_type() != NONE_TOKEN_TYPE);

    // 查看当前child map的大小
    int child_num = this->token_of_child_map.size();

    // 插入一个alphasparse类型的code
    if (child_num > 0)
    {
        assert(this->child_is_exist("AlphaSparse_code_" + to_string(child_num - 1)) == true || this->child_is_exist("raw_cuda_code_" + to_string(child_num - 1)) == true);
    }
    
    // 加入一个元素
    this->token_of_child_map["AlphaSparse_code_" + to_string(child_num)] = alpha_code_line;
}

// 添加一行裸的cuda代码
void reduction_basic_token::add_raw_cuda_code_line(shared_ptr<raw_cuda_code_line> raw_cuda_code_line)
{
    cout << "reduction_basic_token::add_raw_cuda_code_line: this interface is deprecated" << endl;
    assert(false);

    int child_num = this->token_of_child_map.size();

    // 插入一个alphasparse类型的code
    if (child_num > 0)
    {
        assert(this->child_is_exist("AlphaSparse_code_" + to_string(child_num - 1)) == true || this->child_is_exist("raw_cuda_code_" + to_string(child_num - 1)) == true);
    }
    
    // 加入一个元素
    this->token_of_child_map["raw_cuda_code_" + to_string(child_num)] = raw_cuda_code_line;
}

// 直接在这里生成要输出的代码
string reduction_basic_token::run()
{
    assert(this->static_check() == true);

    string return_str =  "// reduction_basic_token: " + convert_pos_type_to_string(this->token_position) + "\n";

    // 遍历并打印所有元素
    for (int i = 0; i < this->token_of_child_map.size(); i++)
    {
        // 输出每一个节点
        if (this->child_is_exist("AlphaSparse_code_" + to_string(i)) == true)
        {
            shared_ptr<basic_token> code_line = this->get_token_of_child("AlphaSparse_code_" + to_string(i));
            return_str = return_str + code_line->run() + "\n";
        }
        else if (this->child_is_exist("raw_cuda_code_" + to_string(i)) == true)
        {
            shared_ptr<basic_token> code_line = this->get_token_of_child("raw_cuda_code_" + to_string(i));
            return_str = return_str + code_line->run() + "\n";
        }
        else
        {
            assert(false);
        }
    }
    
    return return_str;
}

bool reduction_basic_token::static_check()
{
    // 便利所有的子节点
    for (int i = 0; i < this->token_of_child_map.size(); i++)
    {
        // 查看两种子节点那一种存在
        if (this->child_is_exist("AlphaSparse_code_" + to_string(i)) == true)
        {
            // 将当前行取出
            shared_ptr<basic_token> alpha_code_line = this->get_token_of_child("AlphaSparse_code_" + to_string(i));
            assert(alpha_code_line->get_token_type() != RAW_CUDA_CODE_LINE && alpha_code_line->get_token_type() != NONE_TOKEN_TYPE);
            
            string alpha_code_line_str = alpha_code_line->run();

            if (alpha_code_line_str == "" || alpha_code_line_str.size() == 0)
            {
                cout << "reduction_basic_token::static_check: the alpha code is empty" << endl;
                return false;
            }
            
            // 查看结尾是不是有分号
            if (alpha_code_line_str.at(alpha_code_line_str.size() - 1) != ';')
            {
                cout << "reduction_basic_token::static_check: the alpha code is not ended by \';\', not a complete line" << endl;
                return false;
            }

            // 递归静态检查
            if (alpha_code_line->static_check() == false)
            {
                cout << "reduction_basic_token::static_check: invaild alpha_code_line" << endl;
                return false;
            }
        }
        else if (this->child_is_exist("raw_cuda_code_" + to_string(i)) == true)
        {
            // 将当前行取出
            shared_ptr<raw_cuda_code_line> raw_code_line = dynamic_pointer_cast<raw_cuda_code_line>(this->get_token_of_child("raw_cuda_code_" + to_string(i)));
            assert(raw_code_line->get_token_type() == RAW_CUDA_CODE_LINE);

            string raw_code_line_str = raw_code_line->run();

            if (raw_code_line_str == "" || raw_code_line_str.size() == 0)
            {
                cout << "reduction_basic_token::static_check: the raw code is empty" << endl;
                return false;
            }
            
            // 查看结尾是不是有分号
            if (raw_code_line_str.at(raw_code_line_str.size() - 1) != ';')
            {
                cout << "reduction_basic_token::static_check: the raw code is not ended by \';\', not a complete line" << endl;
                return false;
            }

            // 递归检查
            if (raw_code_line->static_check() == false)
            {
                cout << "reduction_basic_token::static_check: invalid raw_code_line" << endl;
                return false;
            }
        }
        else
        {
            assert(false);
        }
    }

    return true;
}