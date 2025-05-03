#include "../kernel_generator.h"

metadata_set_get_token::metadata_set_get_token(POS_TYPE pos)
    : metadata_get_basic_token(pos)
{
    // 实际的构造函数没有东西
    num_of_var_init = 0;
}

void metadata_set_get_token::add_metadata_get_expr(shared_ptr<var_init_token> init_expr)
{
    assert(init_expr != NULL);
    assert(init_expr->get_token_type() == VAR_INIT_TOKEN_TYPE);

    this->token_of_child_map["init_expr_" + to_string(this->num_of_var_init)] = init_expr;

    this->num_of_var_init++;
}

void metadata_set_get_token::add_metadata_get_expr(shared_ptr<var_assign_token> metadata_get_expr)
{
    assert(metadata_get_expr != NULL);
    assert(metadata_get_expr->get_token_type() == VAR_ASSIGN_TOKEN_TYPE);

    this->token_of_child_map["metadata_get_expr_" + to_string(this->num_of_metadata_get_expr)] = metadata_get_expr;

    this->num_of_metadata_get_expr++;
}

void metadata_set_get_token::add_metadata_get_expr(shared_ptr<arr_access_token> metadata_get_expr)
{
    assert(metadata_get_expr != NULL);
    assert(metadata_get_expr->get_token_type() == ARR_ACCESS_TOKEN_TYPE);

    this->token_of_child_map["metadata_get_expr_" + to_string(this->num_of_metadata_get_expr)] = metadata_get_expr;

    this->num_of_metadata_get_expr++;
}

void metadata_set_get_token::add_metadata_get_expr(shared_ptr<shared_mem_broadcast_token> metadata_get_expr)
{
    assert(metadata_get_expr != NULL);
    assert(metadata_get_expr->get_token_type() == SHARED_MEM_BROADCAST_TOKEN_TYPE);

    this->token_of_child_map["metadata_get_expr_" + to_string(this->num_of_metadata_get_expr)] = metadata_get_expr;

    this->num_of_metadata_get_expr++;
}

void metadata_set_get_token::add_metadata_get_expr(shared_ptr<basic_token> metadata_get_expr)
{
    assert(metadata_get_expr != NULL);
    // assert(metadata_get_expr->get_token_type() == SHARED_MEM_BROADCAST_TOKEN_TYPE || metadata_get_expr->get_token_type() == ARR_ACCESS_TOKEN_TYPE ||
    //        metadata_get_expr->get_token_type() == VAR_ASSIGN_TOKEN_TYPE || metadata_get_expr->get_token_type() == VAR_INIT_TOKEN_TYPE );
    
    // this->token_of_child_map["metadata_get_expr_" + to_string(this->num_of_metadata_get_expr)] = metadata_get_expr;

    // 根据类型查看应该怎么自增
    if (metadata_get_expr->get_token_type() == VAR_INIT_TOKEN_TYPE)
    {
        this->token_of_child_map["init_expr_" + to_string(this->num_of_var_init)] = metadata_get_expr;
        this->num_of_var_init++;
    }
    else
    {
        this->token_of_child_map["metadata_get_expr_" + to_string(this->num_of_metadata_get_expr)] = metadata_get_expr;
        this->num_of_metadata_get_expr++;
    }
}

void metadata_set_get_token::add_special_assign_expr(shared_ptr<basic_token> assign_expr)
{

    this->token_of_child_map["assign_expr" + to_string(this->num_of_special_assign_expr)] = assign_expr;
    this->num_of_special_assign_expr++;
    
}

// 执行当前的metadata get
string metadata_set_get_token::run()
{
    assert(this->static_check() == true);

    string return_str = "";

    for (int i = 0; i < num_of_var_init; i++)
    {
        // 首先找出来新的初始化
        assert(this->child_is_exist("init_expr_" + to_string(i)) == true);
        // 执行当前的内容
        return_str = return_str + this->token_of_child_map["init_expr_" + to_string(i)]->run() + "\n";
    }

    for (int i = 0; i < num_of_metadata_get_expr; i++)
    {
        assert(this->child_is_exist("metadata_get_expr_" + to_string(i)) == true);
        return_str = return_str + this->token_of_child_map["metadata_get_expr_" + to_string(i)]->run() + "\n";
    }


    
    for (int i = 0; i < num_of_special_assign_expr; i++)
    {
        assert(this->child_is_exist("assign_expr_" + to_string(i)) == true);
        return_str = return_str + this->token_of_child_map["assign_expr_" + to_string(i)]->run() + "\n";
    }

    return return_str;
}

// 静态检查
bool metadata_set_get_token::static_check()
{
    // 首先遍历所有的初始化代码
    for (int i = 0; i < num_of_var_init; i++)
    {
        assert(this->child_is_exist("init_expr_" + to_string(i)) == true);
        // 将内容取出来
        shared_ptr<var_init_token> var_init_token_ptr = dynamic_pointer_cast<var_init_token>(this->token_of_child_map["init_expr_" + to_string(i)]);
        assert(var_init_token_ptr->get_token_type() == VAR_INIT_TOKEN_TYPE);

        // 递归检查
        if (var_init_token_ptr->static_check() == false)
        {
            cout << "metadata_set_get_token::static_check(): invalid init_expr_" << i << endl;
            return false;
        }
    }

    // 其他赋值的代码
    for (int i = 0; i < num_of_metadata_get_expr; i++)
    {
        assert(this->child_is_exist("metadata_get_expr_" + to_string(i)) == true);
        // 将内容提取出来
        shared_ptr<basic_token> metadata_get_token_ptr = this->token_of_child_map["metadata_get_expr_" + to_string(i)];
        // assert(metadata_get_token_ptr->get_token_type() == ARR_ACCESS_TOKEN_TYPE || metadata_get_token_ptr->get_token_type() == VAR_ASSIGN_TOKEN_TYPE ||
        //        metadata_get_token_ptr->get_token_type() == SHARED_MEM_BROADCAST_TOKEN_TYPE);

        // 对于共享内存广播来说，只有metadata是tblock类型的
        if (metadata_get_token_ptr->get_token_type() == SHARED_MEM_BROADCAST_TOKEN_TYPE && this->get_token_position() != TBLOCK_META)
        {
            cout << "metadata_set_get_token::static_check(): SHARED_MEM_BROADCAST_TOKEN_TYPE can only be used in TBLOCK_META loop, this->get_token_position():" << convert_pos_type_to_string(this->get_token_position()) << endl;
            return false;
        }
        
        // 递归检查
        if (metadata_get_token_ptr->static_check()== false)
        {
            cout << "metadata_set_get_token::static_check(): invalid metadata_get_expr_" << i << endl;
            return false;
        }        
    }

    return true;
}