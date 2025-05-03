#include "../kernel_generator.h"
#include "../data_transform_common.hpp"

// 非终结符
for_basic_token::for_basic_token(POS_TYPE token_position, shared_ptr<metadata_get_basic_token> metadata_get_code, shared_ptr<for_basic_token> inner_loop, shared_ptr<reduction_basic_token> reduction_code, shared_ptr<basic_glue_code> glue_code_block)
:basic_token(false, FOR_BASIC_TOKEN_TYPE)
{
    assert(check_pos_type(token_position) == true);
    assert(metadata_get_code != NULL);
    assert(token_position == TBLOCK_META || token_position == WARP_META || token_position == THREAD_META);

    // 读取元数据，对于内部循环和归约是空的就不用读
    this->token_of_child_map["metadata_get_code"] = metadata_get_code;
    
    if (inner_loop != NULL)
    {
        this->token_of_child_map["inner_loop"] = inner_loop;
    }
    
    if (reduction_code != NULL)
    {
        this->token_of_child_map["reduction_code"] = reduction_code;
    }

    // 加入胶水代码
    if (glue_code_block != NULL)
    {
        // 有归约的时候才可能有胶水代码
        assert(reduction_code != NULL);
        this->token_of_child_map["glue_code_block"] = glue_code_block;
    }

    this->token_position = token_position;
}

string for_basic_token::for_header_run()
{
    assert(this->static_check() == true);
    
    string return_str = "for : " + convert_pos_type_to_string(this->token_position);

    return return_str;
}

string for_basic_token::for_body_run()
{
    assert(this->static_check() == true);
    assert(this->child_is_exist("metadata_get_code") == true);

    string return_str = this->token_of_child_map["metadata_get_code"]->run() + "\n";
    
    // 如果有对应的内部循环，才执行，for循环自带回车，所以不需要额外加回车
    if (this->child_is_exist("inner_loop"))
    {
        return_str = return_str + this->token_of_child_map["inner_loop"]->run();
    }

    // 如果有归约才有对应输出
    if (this->child_is_exist("reduction_code"))
    {
        return_str = return_str + this->token_of_child_map["reduction_code"]->run() + "\n";

        // 有胶水就添加胶水
        if (this->child_is_exist("glue_code_block"))
        {

            assert(this->token_of_child_map["glue_code_block"] != NULL);
            return_str = return_str + this->token_of_child_map["glue_code_block"]->run() + "\n";
        }
    }

    return return_str;
}

// 首先输出for循环头部以及内部的内容
string for_basic_token::run()
{
    assert(this->static_check() == true);
    assert(this->child_is_exist("metadata_get_code") == true);

    string return_str = this->for_header_run() + "\n";

    return_str = return_str + "{\n";

    return_str = return_str + this->for_body_run();

    return_str = return_str + "}\n";

    return return_str;
}

bool for_basic_token::static_check()
{
    assert(this->child_is_exist("metadata_get_code") == true);

    // 严格来说没有什么需要检查的，执行递归检查
    if (this->child_is_exist("metadata_get_code") == true)
    {
        shared_ptr<metadata_get_basic_token> metadata_get_code = dynamic_pointer_cast<metadata_get_basic_token>(this->token_of_child_map["metadata_get_code"]);
        assert(metadata_get_code->get_token_type() == METADATA_GET_BASIC_TOKEN_TYPE);

        if (metadata_get_code->static_check() == false)
        {
            cout << "for_basic_token::static_check(): invalid metadata_get_code" << endl;
            return false;
        }

        // 查看pos类型是不是和当前for循环的pos类型一致
        if (metadata_get_code->get_token_position() != this->token_position)
        {
            cout << "for_basic_token::static_check(): invalid token position of metadata_get_code. metadata_get_code->get_token_position():" << convert_pos_type_to_string(metadata_get_code->get_token_position()) << ", this->token_position:" << convert_pos_type_to_string(this->token_position) << endl;
            return false;
        }
    }

    if (this->child_is_exist("inner_loop"))
    {
        shared_ptr<for_basic_token> inner_loop = dynamic_pointer_cast<for_basic_token>(this->token_of_child_map["inner_loop"]);
        assert(inner_loop->get_token_type() == FOR_BASIC_TOKEN_TYPE || inner_loop->get_token_type() == FOR_TOKEN_TYPE);

        if (inner_loop->static_check() == false)
        {
            cout << "for_basic_token::static_check(): invalid inner_loop" << endl;
            return false;
        }

        // 查看当前的POS类型是不是在自己内部
        if (!(former_pos_is_smaller_than_latter(inner_loop->get_token_position(), this->token_position)))
        {
            cout << "for_basic_token::static_check(): invalid token position of inner_loop. this->token_position:" << convert_pos_type_to_string(this->token_position) << ", inner_loop->get_token_position():" << convert_pos_type_to_string(inner_loop->get_token_position()) << endl;
            return false;
        }
    }

    if (this->child_is_exist("reduction_code"))
    {
        shared_ptr<reduction_basic_token> reduction_code = dynamic_pointer_cast<reduction_basic_token>(this->token_of_child_map["reduction_code"]);
        assert(reduction_code->get_token_type() == REDUCTION_BASIC_TOKEN_TYPE);

        if (reduction_code->static_check() == false)
        {
            cout << "for_basic_token::static_check(): invalid reduction_code" << endl;
            return false;
        }

        if (reduction_code->get_token_position() != this->token_position)
        {
            cout << "for_basic_token::static_check(): invalid token postion of reduction_code. reduction_code->get_token_position():" << convert_pos_type_to_string(reduction_code->get_token_position()) << ", this->token_position:" << convert_pos_type_to_string(this->token_position) << endl;
            return false;
        }
    }

    return true;
}

shared_ptr<for_basic_token> for_basic_token::get_child_for_token()
{
    // 查看是不是存在子索引
    if (this->child_is_exist("inner_loop") == true)
    {
        shared_ptr<for_basic_token> inner_loop_ptr = dynamic_pointer_cast<for_basic_token>(this->token_of_child_map["inner_loop"]);
        assert(inner_loop_ptr != NULL);
        assert(inner_loop_ptr->static_check() == true);
        return inner_loop_ptr;
    }

    return NULL;
}

shared_ptr<reduction_basic_token> for_basic_token::get_reduction_token()
{
    // 查看是不是存在对应的归约
    if (this->child_is_exist("reduction_code") == true)
    {
        shared_ptr<reduction_basic_token> reduction_ptr = dynamic_pointer_cast<reduction_basic_token>(this->token_of_child_map["reduction_code"]);
        assert(reduction_ptr != NULL);
        assert(reduction_ptr->static_check() == true);
        return reduction_ptr;
    }

    return NULL;
}

shared_ptr<metadata_get_basic_token> for_basic_token::get_metadata_get_token()
{
    // 查看是不是存在对应的元数据获取token
    if (this->child_is_exist("metadata_get_code") == true)
    {
        shared_ptr<metadata_get_basic_token> metadata_get_ptr = dynamic_pointer_cast<metadata_get_basic_token>(this->token_of_child_map["metadata_get_code"]);
        assert(metadata_get_ptr != NULL);
        assert(metadata_get_ptr->static_check() == true);
        return metadata_get_ptr;
    }

    return NULL;
}

// 插入token
void for_basic_token::set_metadata_get_code(shared_ptr<metadata_get_basic_token> token_ptr)
{
    assert(token_ptr != NULL);
    assert(token_ptr->static_check() == true);

    this->token_of_child_map["metadata_get_code"] = token_ptr;
}

void for_basic_token::set_reduction_code(shared_ptr<reduction_basic_token> token_ptr)
{
    assert(token_ptr != NULL);
    assert(token_ptr->static_check() == true);

    this->token_of_child_map["reduction_code"] = token_ptr;
}

void for_basic_token::set_glue_code_block(shared_ptr<basic_glue_code> token_ptr)
{
    if(token_ptr == NULL)
    {
        return;
    }
    assert(token_ptr->static_check() == true);

    this->token_of_child_map["glue_code_block"] = token_ptr;
}