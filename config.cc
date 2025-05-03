#include "config.hpp"
configor::json get_config()
{
    // 从外面读配置文件
    configor::json return_json;
    // 处理
    ifstream ifs(CONFIG_FILE_NAME);

    ifs >> return_json;

    ifs.close();

    return return_json;
}


void set_config(string name, string str)
{
    configor::json return_json;
    ifstream ifs(CONFIG_FILE_NAME);
    ifs  >> return_json;
    ifs.close();
    ofstream ofs(CONFIG_FILE_NAME);
    return_json[name] = str;
    ofs << return_json;

    ofs.close();
}

void set_config(string name, int val)
{
    configor::json return_json;
    ifstream ifs(CONFIG_FILE_NAME);
    ifs  >> return_json;
    ifs.close();
    ofstream ofs(CONFIG_FILE_NAME);
    return_json[name] = val;
    ofs << return_json;

    ofs.close();
}