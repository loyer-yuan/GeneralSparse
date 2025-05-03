#include "executor.hpp"
#include <iterator>

using namespace std;

bool execute_binary(string execute_path, string filename, float &total_execute_time, float &gflops)
{
    // execute_path是执行目录
    // 删除执行目录可能存在结果文件
    string command = "cd " + execute_path + " && rm perf_result";
    system(command.c_str());

    // 执行编译
    bool is_compiler_success = compile_spmv_code(execute_path);

    if (is_compiler_success == false)
    {
        return false;
    }

    // 这里代表编译成功，成功之后就开始执行
    command = "cd " + execute_path + " && ./a.out " + get_config()["DATA_SET"].as_string() + filename + "";

    // 执行
    system(command.c_str());

    // 执行完之后查看结果文件是不是写出来了
    ifstream read_perf_result(execute_path + "/perf_result");

    // 有没有对应的文件
    if (!read_perf_result)
    {
        cout << "fail execution" << endl;
        return false;
    }

    float file_time;
    float file_gflops;

    // 获得时间
    read_perf_result >> file_time;
    read_perf_result >> file_gflops;

    // cout << file_time << " " << file_gflops << endl;

    read_perf_result.close();

    total_execute_time = file_time;
    gflops = file_gflops;

    if (file_gflops > get_config()["GFLOPS_UP_BOUND"].as_integer())
    {
        cout << "gflops is too high, maybe some mistake happened in kernal" << endl;
        return false;
    }

    return true;
}

bool compile_spmv_code(string execute_path)
{
    // 删除编译器的输出
    string command = "cd " + execute_path + " && rm compile_result";
    system(command.c_str());

    // 执行编译
    command = "cd " + execute_path + " && sh make_kernel.sh > compile_result 2>&1";
    system(command.c_str());

    // 读取编译结果中的内容，放在compile_result中
    ifstream in(execute_path + "/compile_result");
    istreambuf_iterator<char> begin(in);
    istreambuf_iterator<char> end;
    string compile_result(begin, end);

    string::size_type position = compile_result.find("error");


    // 查看编译的结果
    if (position != compile_result.npos)
    {
        // 编译有错误
        cout << "compile error!" << endl
             << compile_result << endl;
        assert(false);
        return false;
    }
    else
    {
        // 编译没有错误
        cout << "compile success" << endl;
    }

    ifstream fin(execute_path + "/a.out");

    if (!fin)
    {
        cout << "can't find compile result" << endl;
        return false;
    }

    fin.close();

    return true;
}