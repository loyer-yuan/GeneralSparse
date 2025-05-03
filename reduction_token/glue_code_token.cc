#include "../kernel_generator.h"

glue_code_token::glue_code_token(shared_ptr<basic_IO_of_reduction> input_IO, shared_ptr<basic_IO_of_reduction> output_IO, string run_result)
    :basic_glue_code(input_IO, output_IO)
{
    this->run_result = run_result;
}

string glue_code_token::run()
{
    return this->run_result;
}

// TODO检查里面的变量名
bool glue_code_token::static_check()
{
    return true;
}