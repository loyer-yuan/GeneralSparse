#include "../kernel_generator.h"

metadata_get_basic_token::metadata_get_basic_token(POS_TYPE token_position)
:basic_token(false, METADATA_GET_BASIC_TOKEN_TYPE)
{
    // 默认构造函数
    assert(check_pos_type(token_position) == true);
    assert(token_position != NONE_META);
    this->token_position = token_position;
}

string metadata_get_basic_token::run()
{
    return "// metadata_get_basic_token: " + convert_pos_type_to_string(this->token_position);
}

bool metadata_get_basic_token::static_check()
{
    return true;
}