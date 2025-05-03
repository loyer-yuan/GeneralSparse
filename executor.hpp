#ifndef EXECUTOR_H
#define EXECUTOR_H

#include "code_builder.hpp"

// 编译源文件，查看返回值是编译是不是能通过
bool compile_spmv_code(string execute_path);

// 执行代码，并且获得结果，用一个返回值判断是不是执行成功
bool execute_binary(string execute_path, string filename, float& total_execute_time, float& gflops);

#endif