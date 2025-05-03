// 元数据数据库
#ifndef METADATA_SET_HPP
#define METADATA_SET_HPP

#include "code_source_data.hpp"
#include <map>
#include "config.hpp"
#include <unistd.h>

using namespace std;

// metadata所处的位置
enum POS_TYPE
{
    GLOBAL_META = -99,
    TBLOCK_META,
    WARP_META,
    THREAD_META,
    ROW_META,
    COL_META,
    VAL_META,
    NONE_META,
};

// 打印POS_TYPE的类型
string convert_pos_type_to_string(POS_TYPE type);
// 查看类型是不是正确
bool check_pos_type(POS_TYPE meta_position);
// 查看pos类型的优先级，主要是GLOBAL到THREAD的优先级
int priority_of_pos_type(POS_TYPE meta_position);
// 检查并行并行级别的父子关系，前者是不是后者的父并行级别
bool former_is_parent_of_latter(POS_TYPE former, POS_TYPE latter);

// 查看元数据是常量还是数组，用来区别一个元数据是数组还是常量。因为本质上都是universal_array，但是长度为1的数组和常量要区分对待。
enum META_TYPE
{
    ARR_META_TYPE = -80,
    CON_META_TYPE,
    NONE_META_TYPE,
};

// 打印META_TYPE的类型
string convert_meta_type_to_string(META_TYPE type);
// 查看元数据的类型是不是合法
bool check_meta_type(META_TYPE type);

//通过hash，使得switch能在string上使用
typedef std::uint64_t hash_t;
constexpr hash_t prime = 0x100000001B3ull;
constexpr hash_t basis = 0xCBF29CE484222325ull;

hash_t hash_(string str);

constexpr hash_t hash_compile_time(char const *str, hash_t last_value = basis);

constexpr unsigned long long operator"" _hash(char const *p, size_t);

// 通过一个索引的位置，真名，以及所属的子矩阵，获得Metadata set中一个表项的名字
string get_metadata_item_name(POS_TYPE meta_position, string name, int sub_matrix_id);

// 在metadata set中的每一个表项，包含了每一个数据
class meta_data_item
{
public:
    // 使用一个数组指针初始化，接收universal_array数组初始化
    meta_data_item(shared_ptr<universal_array> meta_data_arr, POS_TYPE meta_position, string name, int sub_matrix_id);

    meta_data_item(void *meta_data_ptr, data_type meta_data_type, POS_TYPE meta_position, string name, int sub_matrix_id);

    // 将每一个条目的内容输出到文件中
    void output_2_file(string dest_file);

    // 查看一条metadata是不是正确
    bool check();

    // 获得具体的通用数组
    shared_ptr<universal_array> get_metadata_arr()
    {
        return meta_data_arr;
    }

    // ~meta_data_item()
    // {
    //     cout << "meta_data_item:" << name << " is deleted." << endl;
    // }

    // 用一个bool值判断其是不是加入了format
    bool needed_by_format = false;
    // 当前元素所处的位置
    POS_TYPE meta_position = NONE_META;
    // 当前的matadata的名字
    string name = "none";
    // 查看云数据的类型，数组还是常量
    META_TYPE metadata_type = NONE_META_TYPE;
    // 查看当前数组所在的子矩阵，最终会变成变量名的后缀，-1代表是未分块的全局原始矩阵的基本信息
    int sub_matrix_id = -1;

private:
    // 有一个数组
    shared_ptr<universal_array> meta_data_arr = NULL;
};

// 一个存储meta_data_item的kv数据库，名字是变量名，name和sub_matrix_id的组合："name_sub_matrix_id"
class meta_data_set
{
public:
    // 增加一个元素
    void add_element(shared_ptr<meta_data_item> meta_data_item_ptr);

    // 删除一个元素
    void remove_element(string item_name);
    void remove_element(POS_TYPE type, string real_name, int sub_matrix_id);

    // 获得一个元素的指针
    shared_ptr<meta_data_item> get_element(string item_name);

    // 通过数据所处的循环嵌套的位置和矩阵的id
    shared_ptr<meta_data_item> get_element(POS_TYPE type, string real_name, int sub_matrix_id);

    // 检查所有的元素
    bool check();

    //逻辑检查
    friend bool logical_check(shared_ptr<meta_data_set> meta_data_set_ptr);

    // 打印整个metaset，输出到一个文件夹中，并且输出的东西也是一个名字为随机数的文件夹，内部包含了一系列文件，文件名为key值
    void output_2_dir(string dest_dir_name = (get_config()["ROOT_PATH_STR"].as_string() + "/data_source"));

    // 查看一个name对应的结果是不是存在
    bool is_exist(string item_name);

    // 统计子矩阵的数量
    int get_submatrix_num();

    // 查看
    bool is_exist(POS_TYPE type, string real_name, int sub_matrix_id);

    string matrix_name = "";

    // 获取某一个数据的所能在的最大子矩阵号
    int get_max_sub_matrix_id_of_data_item(POS_TYPE type, string real_name);

    // 查看某个子矩阵各个阶段的节点数量
    int count_of_metadata_of_diff_pos(POS_TYPE type, int sub_matrix_id);

    vector<string> all_item_of_metadata_of_diff_pos(POS_TYPE type, int sub_matrix_id);

    // 将需要的数据放到硬盘中，返回是一个hash值，代表实际存储的位置
    // 存储的位置在配置中的ROOT_PATH_STR界定
    unsigned long output_format_to_dir(vector<POS_TYPE> pos_of_needed_metadata_vec, vector<string> real_name_of_needed_metadata_vec, vector<int> sub_matrix_id_of_needed_metadata_vec);

private:
    map<string, shared_ptr<meta_data_item>> data_map;
};

// 将一个mtx.coo文件放到metadata中
shared_ptr<meta_data_set> create_init_metadata_set_from_file(string file_name, string matrix_name, string float_precise = get_config()["PRECISE_OF_FLOAT"].as_string());
shared_ptr<meta_data_set> create_init_metadata_set_from_file_int(string file_name, string matrix_name, string float_precise = get_config()["PRECISE_OF_FLOAT"].as_string());

// 查看一个文件是不是存在
bool file_is_exist(string file_name);

#endif