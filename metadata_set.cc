#include "metadata_set.hpp"
#include "data_transform_common.hpp"
#include <string.h>
#include <stdio.h>
#include <algorithm>


string convert_pos_type_to_string(POS_TYPE type)
{
    assert(check_pos_type(type));

    if (type == GLOBAL_META)
    {
        return "GLOBAL_META";
    }

    if (type == TBLOCK_META)
    {
        return "TBLOCK_META";
    }

    if (type == WARP_META)
    {
        return "WARP_META";
    }

    if (type == THREAD_META)
    {
        return "THREAD_META";
    }

    if (type == ROW_META)
    {
        return "ROW_META";
    }

    if (type == COL_META)
    {
        return "COL_META";
    }

    if (type == VAL_META)
    {
        return "VAL_META";
    }

    if (type == NONE_META)
    {
        return "NONE_META";
    }

    assert(false);
    return "";
}

bool check_pos_type(POS_TYPE meta_position)
{
    if (meta_position == GLOBAL_META || meta_position == TBLOCK_META || meta_position == WARP_META || meta_position == THREAD_META || meta_position == COL_META ||
        meta_position == VAL_META || meta_position == ROW_META || meta_position == NONE_META)
    {
        return true;
    }

    cout << "invalid positon meta_position:" << meta_position << endl;

    return false;
}

int priority_of_pos_type(POS_TYPE meta_position)
{
    assert(meta_position == GLOBAL_META || meta_position == TBLOCK_META || meta_position == WARP_META || meta_position == THREAD_META);

    if (meta_position == GLOBAL_META)
    {
        return 4;
    }

    if (meta_position == TBLOCK_META)
    {
        return 3;
    }

    if (meta_position == WARP_META)
    {
        return 2;
    }

    if (meta_position == THREAD_META)
    {
        return 1;
    }

    assert(false);
    return 0;
}

bool former_is_parent_of_latter(POS_TYPE former, POS_TYPE latter)
{
    assert(former == GLOBAL_META || former == TBLOCK_META || former == WARP_META || former == THREAD_META);

    int former_priority = priority_of_pos_type(former);
    int latter_priority = priority_of_pos_type(latter);
    
    if (former_priority > latter_priority)
    {
        return true;
    }

    return false;
}

string convert_meta_type_to_string(META_TYPE type)
{
    assert(check_meta_type(type));

    if (type == ARR_META_TYPE)
    {
        return "ARR_META_TYPE";
    }

    if (type == CON_META_TYPE)
    {
        return "CON_META_TYPE";
    }

    if (type == NONE_META_TYPE)
    {
        return "NONE_META_TYPE";
    }

    assert(false);
    return "";
}

bool check_meta_type(META_TYPE type)
{
    if (type == ARR_META_TYPE || type == CON_META_TYPE || type == NONE_META_TYPE)
    {
        return true;
    }

    cout << "invalid type meta_type:" << type << endl;

    return false;
}

string get_metadata_item_name(POS_TYPE meta_position, string name, int sub_matrix_id)
{
    string metadata_item_name = convert_pos_type_to_string(meta_position) + "_" + name + "_" + to_string(sub_matrix_id);
    return metadata_item_name;
}

meta_data_item::meta_data_item(shared_ptr<universal_array> meta_data_arr, POS_TYPE meta_position, string name, int sub_matrix_id)
{
    assert(meta_data_arr != NULL);
    assert(meta_data_arr->check());
    assert(check_pos_type(meta_position) && meta_position != NONE_META);
    assert(name != "");
    assert(sub_matrix_id >= -1);

    // 对元数据进行赋值
    this->meta_data_arr = meta_data_arr;
    this->meta_position = meta_position;
    this->name = name;
    this->metadata_type = ARR_META_TYPE;
    this->sub_matrix_id = sub_matrix_id;

    assert(this->check());
}

meta_data_item::meta_data_item(void *meta_data_ptr, data_type meta_data_type, POS_TYPE meta_position, string name, int sub_matrix_id)
{
    assert(meta_data_ptr != NULL);
    assert(check_pos_type(meta_position) && meta_position != NONE_META);
    assert(meta_data_type == UNSIGNED_CHAR || meta_data_type == UNSIGNED_SHORT || meta_data_type == UNSIGNED_INT || meta_data_type == UNSIGNED_LONG || meta_data_type == BOOL ||
           meta_data_type == FLOAT || meta_data_type == DOUBLE);
    assert(name != "");
    assert(sub_matrix_id >= -1);

    // 首先创造一个只有一位的universal_array
    shared_ptr<universal_array> single_ele_arr(new universal_array(meta_data_ptr, 1, meta_data_type));
    assert(single_ele_arr->check());

    this->meta_data_arr = single_ele_arr;
    this->meta_position = meta_position;
    this->name = name;
    this->metadata_type = CON_META_TYPE;
    this->sub_matrix_id = sub_matrix_id;

    assert(this->check());
}

void meta_data_item::output_2_file(string dest_file)
{
    // 完成检查
    assert(this->check());

    // cout << "meta_data_item::output_2_file: write to disk: " << dest_file << endl;

    // 打印一个item的内容
    this->meta_data_arr->output_2_file(dest_file);
}

bool meta_data_item::check()
{
    // 做一些检查
    if (this->meta_data_arr == NULL)
    {
        cout << "meta_data_item::check():meta_data_arr is empty ptr" << endl;
        return false;
    }

    // 数组的位置
    if (!(check_pos_type(this->meta_position) && this->meta_position != NONE_META))
    {
        cout << "meta_data_item::check(): meta_position type error" << endl;
        return false;
    }

    if (this->name == "" || this->name == "none")
    {
        cout << "meta_data_item::check(): name error, this->name = " << this->name << endl;
        return false;
    }

    // 元数据的类型，数组或者常量
    if (!(this->metadata_type == CON_META_TYPE || this->metadata_type == ARR_META_TYPE))
    {
        cout << "meta_data_item::check(): metadata_type error" << endl;
        return false;
    }

    // 如果数据长度为1，那就只能是常量
    if (this->meta_data_arr->get_len() > 1 && this->metadata_type == CON_META_TYPE)
    {
        cout << "meta_data_item::check(): metadata_type does not match with array length, this->meta_data_arr->get_len() = " << this->meta_data_arr->get_len() << endl;
        return false;
    }

    if (this->meta_data_arr->check() == false)
    {
        cout << "meta_data_item::check(): can not pass the check of meta_data_arr" << endl;
        return false;
    }

    if (this->sub_matrix_id < -1)
    {
        cout << "meta_data_item::check(): sub_matrix_id is less than -1, sub_matrix_id = " << this->sub_matrix_id << endl;
        return false;
    }

    return true;
}

void meta_data_set::add_element(shared_ptr<meta_data_item> meta_data_item_ptr)
{
    assert(meta_data_item_ptr != NULL);
    assert(meta_data_item_ptr->check());

    // 构建名字的时候，包含了元数据的层次和元数据所属的子矩阵号
    string meta_data_name = get_metadata_item_name(meta_data_item_ptr->meta_position, meta_data_item_ptr->name, meta_data_item_ptr->sub_matrix_id);
    // string meta_data_name = convert_pos_type_to_string(meta_data_item_ptr->meta_position) + "_" + meta_data_item_ptr->name + "_" + to_string(meta_data_item_ptr->sub_matrix_id);

    // 如果之前存在这个key，那就发生了错误
    if (this->data_map.count(meta_data_name) != 0)
    {
        cout << "meta_data_set::add_element: The key \"" + meta_data_name + "\" has existed" << endl;
        assert(false);
    }

    // 将内容带入到meta_data_set中
    this->data_map[meta_data_name] = meta_data_item_ptr;
}

void meta_data_set::remove_element(string item_name)
{
    // 当前键值在不在，不在会导致错误
    if (this->data_map.count(item_name) == 0)
    {
        cout << "meta_data_set::remove_element: The key \"" + item_name + "\" has not existed" << endl;
        assert(false);
    }

    // 执行删除
    this->data_map.erase(item_name);
}

void meta_data_set::remove_element(POS_TYPE type, string real_name, int sub_matrix_id)
{
    string item_name = get_metadata_item_name(type, real_name, sub_matrix_id);
    // string item_name = convert_pos_type_to_string(type) + "_" + real_name + "_" + to_string(sub_matrix_id);
    this->remove_element(item_name);
}

shared_ptr<meta_data_item> meta_data_set::get_element(string item_name)
{
    // 当前键值不存在，那就会导致错误
    if (this->data_map.count(item_name) == 0)
    {
        cout << "meta_data_set::get_element: The key \"" + item_name + "\" has not existed" << endl;
        assert(false);
    }

    // 将一个数据从已有的metadata set中读出来
    return this->data_map[item_name];
}

shared_ptr<meta_data_item> meta_data_set::get_element(POS_TYPE type, string real_name, int sub_matrix_id)
{
    string item_name = get_metadata_item_name(type, real_name, sub_matrix_id);
    // string item_name = convert_pos_type_to_string(type) + "_" + real_name + "_" + to_string(sub_matrix_id);
    return get_element(item_name);
}

bool meta_data_set::is_exist(string item_name)
{
    // 查看当前键值是不是存在
    if (this->data_map.count(item_name) == 0)
    {
        return false;
    }

    return true;
}

bool meta_data_set::is_exist(POS_TYPE type, string real_name, int sub_matrix_id)
{
    string item_name = get_metadata_item_name(type, real_name, sub_matrix_id);
    // string item_name = convert_pos_type_to_string(type) + "_" + real_name + "_" + to_string(sub_matrix_id);
    return is_exist(item_name);
}

// 检查
bool meta_data_set::check()
{
    // 检查所有的元素，申请一个迭代器
    map<string, shared_ptr<meta_data_item>>::iterator iter;
    iter = this->data_map.begin();

    while (iter != this->data_map.end())
    {
        // 检查key和value的命名是不是一致
        // if (iter->first != (convert_pos_type_to_string(iter->second->meta_position) + "_" + iter->second->name + "_" + to_string(iter->second->sub_matrix_id)))
        if (iter->first != get_metadata_item_name(iter->second->meta_position, iter->second->name, iter->second->sub_matrix_id))
        {
            cout << "meta_data_set::check(): name invaild, key:" << iter->first << ", meta_data_item->name:" << iter->second->name << ", sub_matrix_id" << iter->second->sub_matrix_id << endl;
            return false;
        }

        // 检查meta_data_item自身
        if (iter->second->check() == false)
        {
            cout << "meta_data_set::check(): meta_data_item invaild" << endl;
            return false;
        }

        iter++;
    }

    return true;
}

int meta_data_set::get_max_sub_matrix_id_of_data_item(POS_TYPE type, string real_name)
{
    assert(this->check());
    // 遍历所有元素
    map<string, shared_ptr<meta_data_item>>::iterator iter;
    iter = this->data_map.begin();

    // 如果对应的位置和数据名字没有数据，就会直接错误
    bool is_found = false;

    // 需要查找的目标
    string target_prefix = convert_pos_type_to_string(type) + "_" + real_name + "_";

    int max_sub_matrix_id = -1;
    // cout << "1" << endl;
    while (iter != this->data_map.end())
    {
        // 获得当前的key的头部类型
        string key_str = iter->first;

        // 看看能不能搜到
        if (key_str.find(target_prefix) != string::npos)
        {
            // 将头部减掉，获得最后的字符串
            int target_str_len = strlen(target_prefix.c_str());
            int key_str_len = strlen(key_str.c_str());
            assert(target_str_len < key_str_len);

            // 查看剩下的字符串的长度
            int length = key_str_len - target_str_len;
            assert(target_str_len > 0);
            string sub_matrix_id_str = key_str.substr(target_str_len, length);

            // cout << "sub_matrix_id_str:" << sub_matrix_id_str << endl;

            // 将末尾的子矩阵编号换成int类型
            int cur_sub_matrix_id = atoi(sub_matrix_id_str.c_str());

            // cout << "cur_sub_matrix_id:" << cur_sub_matrix_id << endl;

            if (max_sub_matrix_id < cur_sub_matrix_id)
            {
                max_sub_matrix_id = cur_sub_matrix_id;
            }

            is_found = true;
        }

        iter++;
    }

    if (is_found == false)
    {
        cout << "meta_data_set::get_max_sub_matrix_id_of_data_item: not find item,type:" << convert_pos_type_to_string(type) << ",real_name:"
             << real_name << ",target_prefix:" << target_prefix << "," << endl;

        assert(false);
    }

    // 找到了这类数据所在的最大子矩阵号
    return max_sub_matrix_id;
}

int meta_data_set::count_of_metadata_of_diff_pos(POS_TYPE type, int sub_matrix_id)
{
    // 检查当前内容
    assert(this->check());

    // 遍历所有元素
    map<string, shared_ptr<meta_data_item>>::iterator iter;
    iter = this->data_map.begin();
    int count = 0;

    while (iter != this->data_map.end())
    {
        // 搜索，查看当前节点类型
        POS_TYPE cur_type = iter->second->meta_position;
        int cur_matrix_id = iter->second->sub_matrix_id;

        // 节点类型满足要求计数
        if (type == cur_type && cur_matrix_id == sub_matrix_id)
        {
            count++;
        }

        iter++;
    }

    return count;
}

vector<string> meta_data_set::all_item_of_metadata_of_diff_pos(POS_TYPE type, int sub_matrix_id)
{
    // 检查当前内容
    assert(this->check());
    vector<string> item_name;
    // 遍历所有元素
    map<string, shared_ptr<meta_data_item>>::iterator iter;
    iter = this->data_map.begin();

    while (iter != this->data_map.end())
    {
        // 搜索，查看当前节点类型
        POS_TYPE cur_type = iter->second->meta_position;
        int sub_id = iter->second->sub_matrix_id;

        // 记录所有的同位置的子矩阵item
        if (type == cur_type && sub_matrix_id == sub_id)
        {
            assert(count(item_name.begin(), item_name.end(), iter->second->name) == 0);
            item_name.push_back(iter->second->name);
        }

        iter++;
    }

    return item_name;
}

void meta_data_set::output_2_dir(string dest_dir_name)
{
    // 检查当前内容
    assert(this->check());

    // 首先创造一个随机数
    srand(time(0));
    unsigned long meta_data_set_id = rand() + time(0) % 1000;
    sleep(1);

    // 首先找到元数据的目录
    string dir_of_data_source = dest_dir_name + "/" + to_string(meta_data_set_id);

    // 根据这一随机数创造一个目录
    system(("mkdir " + dir_of_data_source).c_str());

    cout << "the whole metadata set is stored in " << dir_of_data_source << endl;

    // 然后遍历metadataset的所有数据，将每一个数组放到文件中，文件名为key的名称
    map<string, shared_ptr<meta_data_item>>::iterator iter;
    iter = this->data_map.begin();

    while (iter != this->data_map.end())
    {
        // 获取当前数据项的名字
        string metadata_name = iter->first;
        // 最终要写的文件名
        string input_file_name = dir_of_data_source + "/" + metadata_name;
        // 写文件
        iter->second->output_2_file(input_file_name);

        iter++;
    }
}

unsigned long meta_data_set::output_format_to_dir(vector<POS_TYPE> pos_of_needed_metadata_vec,
                                        vector<string> real_name_of_needed_metadata_vec, vector<int> sub_matrix_id_of_needed_metadata_vec)
{
    // 必然是存在具体的要写入的索引
    // assert(pos_of_needed_metadata_vec.size() != 0);
    // assert(real_name_of_needed_metadata_vec.size() != 0);
    // assert(sub_matrix_id_of_needed_metadata_vec.size() != 0);
    assert(pos_of_needed_metadata_vec.size() == real_name_of_needed_metadata_vec.size() &&
           real_name_of_needed_metadata_vec.size() == sub_matrix_id_of_needed_metadata_vec.size());
    assert(this->check());

    // 首先创造一个随机数
    srand(time(0));
    unsigned long output_id = rand() + time(0) % 1000;
    // 不能是0
    while (output_id == 0)
    {
        sleep(1);
        output_id = rand() + time(0) % 1000;
    }
    sleep(1);

    string dir_of_data_source = get_config()["ROOT_PATH_STR"].as_string() + "/data_source/" + to_string(output_id);
    cout << "meta_data_set::output_format_to_dir:" << "mkdir " + dir_of_data_source << endl;
    system(("mkdir " + dir_of_data_source).c_str());
    sleep(1);

    // 遍历所有需要输出的索引
    for (int i = 0; i < pos_of_needed_metadata_vec.size(); i++)
    {
        // 查看当前索引是不是存在
        if (this->is_exist(pos_of_needed_metadata_vec[i], real_name_of_needed_metadata_vec[i], sub_matrix_id_of_needed_metadata_vec[i]) == false)
        {
            cout << "meta_data_set::output_format_to_dir: the format indice is not existing" << endl;
            cout << "pos:" << convert_pos_type_to_string(pos_of_needed_metadata_vec[i]) << endl;
            cout << "real_name:" << real_name_of_needed_metadata_vec[i] << endl;
            cout << "pos:" << sub_matrix_id_of_needed_metadata_vec[i] << endl;

            assert(false);
        }
        else
        {
            // cout << "meta_data_set::output_format_to_dir: write format indices to disk. pos:" << convert_pos_type_to_string(pos_of_needed_metadata_vec[i]) << ", real_name:" << real_name_of_needed_metadata_vec[i] << ", sub_matrix_id" << sub_matrix_id_of_needed_metadata_vec[i] << endl;
            // 当前索引存在，将其放到对应的文件夹中
            // 获取当前数据项的名字
            string metadata_name = get_metadata_item_name(pos_of_needed_metadata_vec[i], real_name_of_needed_metadata_vec[i], sub_matrix_id_of_needed_metadata_vec[i]);
            // 最终要写的文件名
            string output_file_name = dir_of_data_source + "/" + metadata_name;
            // 写文件
            this->get_element(pos_of_needed_metadata_vec[i], real_name_of_needed_metadata_vec[i], sub_matrix_id_of_needed_metadata_vec[i])->output_2_file(output_file_name);
        }
    }

    return output_id;
}

hash_t hash_(string str)
{
    hash_t ret{basis};
    string::iterator iter = str.begin();
    while (iter != str.end())
    {
        ret ^= *iter;
        ret *= prime;
        iter++;
    }

    return ret;
}

constexpr hash_t hash_compile_time(char const *str, hash_t last_value)
{
    return *str ? hash_compile_time(str + 1, (*str ^ last_value) * prime) : last_value;
}

constexpr unsigned long long operator"" _hash(char const *p, size_t)
{
    return hash_compile_time(p);
}

int meta_data_set::get_submatrix_num()
{
    set<int> submatrix;
    map<string, shared_ptr<meta_data_item>>::iterator iter;
    iter = this->data_map.begin();
    int num = 0;
    while (iter != this->data_map.end())
    {
        submatrix.insert(iter->second->sub_matrix_id);
        iter++;
    }
    num = submatrix.size();
    return num;
}

shared_ptr<meta_data_set> create_init_metadata_set_from_file(string file_name, string matrix_name, string float_precise)
{
    // vector<half> half_val_vec;
    vector<float> float_val_vec;
    vector<double> double_val_vec;
    unsigned long max_col_index;
    unsigned long max_row_index;
    vector<unsigned long> col_index_vec;
    vector<unsigned long> row_index_vec;

    data_type input_type;

    if (float_precise == "float")
    {
        input_type = FLOAT;
    }
    else if (float_precise == "double")
    {
        input_type = DOUBLE;
    }
    // else if (float_precise == "half")
    // {
    //     input_type = HALF;
    // }
    else
    {
        cout << "create_init_metadata_set_from_file: invalid precise, float_precise: " << float_precise << endl;
        assert(false);
    }

    get_matrix_index_and_val_from_file(file_name, row_index_vec, col_index_vec, float_val_vec, double_val_vec, input_type, max_row_index, max_col_index);

    assert((input_type == FLOAT && double_val_vec.size() == 0) || (input_type == DOUBLE && float_val_vec.size() == 0));
    //  || (input_type == HALF && half_val_vec.size() == 0));

    // 创建一个新的metadata set
    shared_ptr<meta_data_set> meta_data_set_ptr(new meta_data_set());
    meta_data_set_ptr->matrix_name = matrix_name;

    // 首先将矩阵的初始行列数量和nnz存起来
    unsigned long row_number = max_row_index + 1;
    unsigned long col_number = max_col_index + 1;
    unsigned long origin_nnz = row_index_vec.size();
    unsigned long begin_index = 0;

    // 将三个元素放到元数据中
    shared_ptr<meta_data_item> row_number_meta_item_ptr(new meta_data_item((void *)(&row_number), UNSIGNED_LONG, GLOBAL_META, "origin_row_num", -1));
    meta_data_set_ptr->add_element(row_number_meta_item_ptr);

    shared_ptr<meta_data_item> col_number_meta_item_ptr(new meta_data_item((void *)(&col_number), UNSIGNED_LONG, GLOBAL_META, "origin_col_num", -1));
    meta_data_set_ptr->add_element(col_number_meta_item_ptr);

    shared_ptr<meta_data_item> nz_number_meta_item_ptr(new meta_data_item((void *)(&origin_nnz), UNSIGNED_LONG, GLOBAL_META, "origin_nnz_num", -1));
    meta_data_set_ptr->add_element(nz_number_meta_item_ptr);

    // 获得当前子块的起始行和起始列，末行和末列，当前只有一个块，所以直接根据行数量和列数量来得到。
    shared_ptr<meta_data_item> begin_row_bound_item_ptr(new meta_data_item((void *)(&begin_index), UNSIGNED_LONG, GLOBAL_META, "begin_row_index", 0));
    meta_data_set_ptr->add_element(begin_row_bound_item_ptr);

    shared_ptr<meta_data_item> begin_col_bound_item_ptr(new meta_data_item((void *)(&begin_index), UNSIGNED_LONG, GLOBAL_META, "begin_col_index", 0));
    meta_data_set_ptr->add_element(begin_col_bound_item_ptr);

    shared_ptr<meta_data_item> end_row_bound_item_ptr(new meta_data_item((void *)(&max_row_index), UNSIGNED_LONG, GLOBAL_META, "end_row_index", 0));
    meta_data_set_ptr->add_element(end_row_bound_item_ptr);

    shared_ptr<meta_data_item> end_col_bound_item_ptr(new meta_data_item((void *)(&max_col_index), UNSIGNED_LONG, GLOBAL_META, "end_col_index", 0));
    meta_data_set_ptr->add_element(end_col_bound_item_ptr);

    // 将行索引、列索引、值取出来
    shared_ptr<universal_array> row_indices_ptr(new universal_array(&(row_index_vec[0]), row_index_vec.size(), UNSIGNED_LONG));
    shared_ptr<meta_data_item> row_indices_meta_item_ptr(new meta_data_item(row_indices_ptr, GLOBAL_META, "nz_row_indices", 0));
    meta_data_set_ptr->add_element(row_indices_meta_item_ptr);

    shared_ptr<universal_array> col_indices_ptr(new universal_array(&(col_index_vec[0]), col_index_vec.size(), UNSIGNED_LONG));
    shared_ptr<meta_data_item> col_indices_meta_item_ptr(new meta_data_item(col_indices_ptr, GLOBAL_META, "nz_col_indices", 0));
    meta_data_set_ptr->add_element(col_indices_meta_item_ptr);

    shared_ptr<universal_array> vals_ptr = NULL;
    // 根据当前的数据类型选择不同的值数组的构造方式
    if (input_type == DOUBLE)
    {
        assert(double_val_vec.size() == row_index_vec.size());
        vals_ptr = make_shared<universal_array>(&(double_val_vec[0]), double_val_vec.size(), input_type);
    }
    else
    {
        assert(input_type == FLOAT);
        assert(float_val_vec.size() == row_index_vec.size());
        vals_ptr = make_shared<universal_array>(&(float_val_vec[0]), float_val_vec.size(), input_type);
    }
    // 将数组的值vals_ptr放到metadataset中
    shared_ptr<meta_data_item> vals_meta_item_ptr(new meta_data_item(vals_ptr, GLOBAL_META, "nz_vals", 0));
    meta_data_set_ptr->add_element(vals_meta_item_ptr);

    return meta_data_set_ptr;
}

shared_ptr<meta_data_set> create_init_metadata_set_from_file_int(string file_name, string matrix_name, string float_precise)
{
    vector<float> float_val_vec;
    vector<double> double_val_vec;
    unsigned long max_col_index;
    unsigned long max_row_index;
    vector<unsigned int> col_index_vec;
    vector<unsigned int> row_index_vec;

    data_type input_type;

    if (float_precise == "float")
    {
        input_type = FLOAT;
    }
    else if (float_precise == "double")
    {
        input_type = DOUBLE;
    }
    else
    {
        cout << "create_init_metadata_set_from_file: invalid precise, float_precise: " << float_precise << endl;
        assert(false);
    }

    get_matrix_index_and_val_from_file_int(file_name, row_index_vec, col_index_vec, float_val_vec, double_val_vec, input_type, max_row_index, max_col_index);

    assert((input_type == FLOAT && double_val_vec.size() == 0) || (input_type == DOUBLE && float_val_vec.size() == 0));

    // 创建一个新的metadata set
    shared_ptr<meta_data_set> meta_data_set_ptr(new meta_data_set());
    meta_data_set_ptr->matrix_name = matrix_name;

    // 首先将矩阵的初始行列数量和nnz存起来
    unsigned long row_number = max_row_index + 1;
    unsigned long col_number = max_col_index + 1;
    unsigned long origin_nnz = row_index_vec.size();
    unsigned long begin_index = 0;

    if(get_config()["FORMAT_OF_MTX"].as_string() == "CSR")
    {
        read_mtx_as_csr_int(meta_data_set_ptr, row_index_vec, max_row_index);
    }

    // 将三个元素放到元数据中
    shared_ptr<meta_data_item> row_number_meta_item_ptr(new meta_data_item((void *)(&row_number), UNSIGNED_LONG, GLOBAL_META, "origin_row_num", -1));
    meta_data_set_ptr->add_element(row_number_meta_item_ptr);

    shared_ptr<meta_data_item> col_number_meta_item_ptr(new meta_data_item((void *)(&col_number), UNSIGNED_LONG, GLOBAL_META, "origin_col_num", -1));
    meta_data_set_ptr->add_element(col_number_meta_item_ptr);

    shared_ptr<meta_data_item> nz_number_meta_item_ptr(new meta_data_item((void *)(&origin_nnz), UNSIGNED_LONG, GLOBAL_META, "origin_nnz_num", -1));
    meta_data_set_ptr->add_element(nz_number_meta_item_ptr);

    // 获得当前子块的起始行和起始列，末行和末列，当前只有一个块，所以直接根据行数量和列数量来得到。
    shared_ptr<meta_data_item> begin_row_bound_item_ptr(new meta_data_item((void *)(&begin_index), UNSIGNED_LONG, GLOBAL_META, "begin_row_index", 0));
    meta_data_set_ptr->add_element(begin_row_bound_item_ptr);

    shared_ptr<meta_data_item> begin_col_bound_item_ptr(new meta_data_item((void *)(&begin_index), UNSIGNED_LONG, GLOBAL_META, "begin_col_index", 0));
    meta_data_set_ptr->add_element(begin_col_bound_item_ptr);

    shared_ptr<meta_data_item> end_row_bound_item_ptr(new meta_data_item((void *)(&max_row_index), UNSIGNED_LONG, GLOBAL_META, "end_row_index", 0));
    meta_data_set_ptr->add_element(end_row_bound_item_ptr);

    shared_ptr<meta_data_item> end_col_bound_item_ptr(new meta_data_item((void *)(&max_col_index), UNSIGNED_LONG, GLOBAL_META, "end_col_index", 0));
    meta_data_set_ptr->add_element(end_col_bound_item_ptr);

    // 将行索引、列索引、值取出来
    shared_ptr<universal_array> row_indices_ptr(new universal_array(&(row_index_vec[0]), row_index_vec.size(), UNSIGNED_INT));
    shared_ptr<meta_data_item> row_indices_meta_item_ptr(new meta_data_item(row_indices_ptr, GLOBAL_META, "nz_row_indices", 0));
    meta_data_set_ptr->add_element(row_indices_meta_item_ptr);

    shared_ptr<universal_array> col_indices_ptr(new universal_array(&(col_index_vec[0]), col_index_vec.size(), UNSIGNED_INT));
    shared_ptr<meta_data_item> col_indices_meta_item_ptr(new meta_data_item(col_indices_ptr, GLOBAL_META, "nz_col_indices", 0));
    meta_data_set_ptr->add_element(col_indices_meta_item_ptr);

    shared_ptr<universal_array> vals_ptr = NULL;
    // 根据当前的数据类型选择不同的值数组的构造方式
    if (input_type == DOUBLE)
    {
        assert(double_val_vec.size() == row_index_vec.size());
        vals_ptr = make_shared<universal_array>(&(double_val_vec[0]), double_val_vec.size(), input_type);
    }
    else
    {
        assert(input_type == FLOAT);
        assert(float_val_vec.size() == row_index_vec.size());
        vals_ptr = make_shared<universal_array>(&(float_val_vec[0]), float_val_vec.size(), input_type);
    }
    // 将数组的值vals_ptr放到metadataset中
    shared_ptr<meta_data_item> vals_meta_item_ptr(new meta_data_item(vals_ptr, GLOBAL_META, "nz_vals", 0));
    meta_data_set_ptr->add_element(vals_meta_item_ptr);

    return meta_data_set_ptr;
}

//检查各个meta data之间的逻辑关系
bool logical_check(shared_ptr<meta_data_set> meta_data_set_ptr)
{

    //保留基本检查的功能
    meta_data_set_ptr->check();

    //统计子矩阵可能的最大数量，包括原始矩阵
    int num = meta_data_set_ptr->get_submatrix_num();

    cout << "meta_data_set::logical_check(): sub matrix num = " << num << endl;
    //存储检查所需对象
    vector<shared_ptr<universal_array>> nz_row_indices(num, nullptr), nz_col_indices(num, nullptr), nz_vals(num, nullptr), begin_row_index(num, nullptr), begin_col_index(num, nullptr), end_row_index(num, nullptr), end_col_index(num, nullptr);
    vector<shared_ptr<universal_array>> first_nz_indices_tblock(num, nullptr), first_nz_indices_warp(num, nullptr), first_nz_indices_thread(num, nullptr);
    vector<shared_ptr<universal_array>> first_row_indices_tblock(num, nullptr), first_row_indices_warp(num, nullptr),first_row_indices_thread(num,nullptr);
    vector<shared_ptr<universal_array>> first_row_indices_relative_to_BMTB_warp(num, nullptr), first_row_indices_relative_to_BMTB_thread(num, nullptr);
    vector<shared_ptr<universal_array>> first_row_indices_relative_to_BMW(num, nullptr);
    vector<shared_ptr<universal_array>> first_nz_indices_relative_to_BMTB_warp(num, nullptr), first_nz_indices_relative_to_BMTB_thread(num, nullptr);
    vector<shared_ptr<universal_array>> first_nz_indices_relative_to_BMW(num, nullptr);
    vector<shared_ptr<universal_array>> first_nz_indices_without_ending(num, nullptr);
    vector<shared_ptr<universal_array>> first_row_indices_thread_without_ending(num, nullptr);
    vector<shared_ptr<universal_array>> first_row_indices_warp_without_ending(num, nullptr);
    vector<shared_ptr<universal_array>> first_row_indices_tblock_without_ending(num, nullptr);
    vector<shared_ptr<universal_array>> BMT_size_of_each_blk_tblock(num, nullptr), BMT_size_of_each_blk_warp(num, nullptr);
    vector<shared_ptr<universal_array>> first_BMT_indices_tblock(num, nullptr), first_BMT_indices_warp(num, nullptr);
    vector<shared_ptr<universal_array>> first_BMW_indices_tblock(num, nullptr);
    vector<shared_ptr<universal_array>> BMT_size_of_each_blk_global(num, nullptr);
    vector<shared_ptr<universal_array>> nz_col_indices_after_interlance_storage(num, nullptr);
    vector<shared_ptr<universal_array>> nz_row_indices_after_interlance_storage(num, nullptr);
    vector<shared_ptr<universal_array>> nz_vals_after_interlance_storage(num, nullptr);

    map<string, shared_ptr<meta_data_item>>::iterator iter;

    iter = meta_data_set_ptr->data_map.begin();
    //遍历data_map,根据name保存metadata
    while (iter != meta_data_set_ptr->data_map.end())
    {
        int index = (iter->second->sub_matrix_id >= 0) ? iter->second->sub_matrix_id : num - 1;
        switch (hash_(iter->second->name))
        {
        case "nz_row_indices"_hash:
            nz_row_indices[index] = iter->second->get_metadata_arr();
            break;
        case "nz_col_indices"_hash:
            nz_col_indices[index] = iter->second->get_metadata_arr();
            break;
        case "nz_vals"_hash:
            nz_vals[index] = iter->second->get_metadata_arr();
            break;
        case "begin_row_index"_hash:
            begin_row_index[index] = iter->second->get_metadata_arr();
            break;
        case "begin_col_index"_hash:
            begin_col_index[index] = iter->second->get_metadata_arr();
            break;
        case "end_row_index"_hash:
            end_row_index[index] = iter->second->get_metadata_arr();
            break;
        case "end_col_index"_hash:
            end_col_index[index] = iter->second->get_metadata_arr();
            break;
        case "first_nz_indices"_hash:
            if (iter->second->meta_position == TBLOCK_META)
            {
                first_nz_indices_tblock[index] = iter->second->get_metadata_arr();
            }
            else if (iter->second->meta_position == WARP_META)
            {
                first_nz_indices_warp[index] = iter->second->get_metadata_arr();
            }
            else if (iter->second->meta_position == THREAD_META)
            {
                first_nz_indices_thread[index] = iter->second->get_metadata_arr();
            }
            else
            {
                cout << "meta_data_set::logical_check():first_nz_indices position invalid:" << iter->second->meta_position << endl;
                return false;
            }
            break;
        case "first_row_indices"_hash:
            if (iter->second->meta_position == TBLOCK_META)
            {
                first_row_indices_tblock[index] = iter->second->get_metadata_arr();
            }
            else if (iter->second->meta_position == WARP_META)
            {
                first_row_indices_warp[index] = iter->second->get_metadata_arr();
            }
            else if (iter->second->meta_position == THREAD_META)
            {
                first_row_indices_thread[index] = iter->second->get_metadata_arr();
            }
            else
            {
                cout << "meta_data_set::logical_check():first_row_indices position invalid:" << iter->second->meta_position << endl;
                return false;
            }
            break;
        case "first_row_indices_relative_to_BMTB"_hash:
            if (iter->second->meta_position == WARP_META)
            {
                first_row_indices_relative_to_BMTB_warp[index] = iter->second->get_metadata_arr();
            }
            else if (iter->second->meta_position == THREAD_META)
            {
                first_row_indices_relative_to_BMTB_thread[index] = iter->second->get_metadata_arr();
            }
            else
            {
                cout << "meta_data_set::logical_check():first_row_indices_relative_to_BMTB position invalid:" << iter->second->meta_position << endl;
                return false;
            }
            break;
        case "first_row_indices_relative_to_BMW"_hash:
            first_row_indices_relative_to_BMW[index] = iter->second->get_metadata_arr();
            break;
            
        case "first_nz_indices_relative_to_BMW"_hash:
            first_nz_indices_relative_to_BMW[index] = iter->second->get_metadata_arr();
            break;

        case "first_nz_indices_relative_to_BMTB"_hash:
            if(iter->second->meta_position == WARP_META)
            {
                first_nz_indices_relative_to_BMTB_warp[index] = iter->second->get_metadata_arr();
            }else if(iter->second->meta_position == THREAD_META)
            {
                first_nz_indices_relative_to_BMTB_thread[index] = iter->second->get_metadata_arr();
            }
            break;
        case "first_nz_indices_without_ending"_hash:
            first_nz_indices_without_ending[index] = iter->second->get_metadata_arr();
            break;
        case "first_row_indices_without_ending"_hash:
            if(iter->second->meta_position == THREAD_META)
            {
                first_row_indices_thread_without_ending[index] = iter->second->get_metadata_arr();
            }
            else if (iter->second->meta_position == WARP_META)
            {
                first_row_indices_warp_without_ending[index] = iter->second->get_metadata_arr();
            }
            else if(iter->second->meta_position == TBLOCK_META)
            {
                first_row_indices_tblock_without_ending[index] = iter->second->get_metadata_arr();
            }
            else
            {
                cout<<"meta_data_set::logical_check():first_row_indices_without_ending position invalid:" << iter->second->meta_position << endl;
            }
            break;
        case "BMT_size_of_each_blk"_hash:
            if (iter->second->meta_position == GLOBAL_META)
            {
                BMT_size_of_each_blk_global[index] = iter->second->get_metadata_arr();
            }
            else if (iter->second->meta_position == TBLOCK_META)
            {
                BMT_size_of_each_blk_tblock[index] = iter->second->get_metadata_arr();
            }
            else if (iter->second->meta_position == WARP_META)
            {
                BMT_size_of_each_blk_warp[index] = iter->second->get_metadata_arr();
            }
            else
            {
                cout << "meta_data_set::logical_check():BMT_size_of_each_blk position invalid:" << iter->second->meta_position << endl;
                return false;
            }
            break;
        case "first_BMT_indices"_hash:
            if (iter->second->meta_position == TBLOCK_META)
            {
                first_BMT_indices_tblock[index] = iter->second->get_metadata_arr();
            }
            else if (iter->second->meta_position == WARP_META)
            {
                first_BMT_indices_warp[index] = iter->second->get_metadata_arr();
            }
            else
            {
                cout << "meta_data_set::logical_check():first_BMT_indices position invalid:" << iter->second->meta_position << endl;
                return false;
            }
            break;
        case "first_BMW_indices"_hash:
            if (iter->second->meta_position == TBLOCK_META)
            {
                first_BMW_indices_tblock[index] = iter->second->get_metadata_arr();
            }
            else
            {
                cout << "meta_data_set::logical_check():first_BMT_indices position invalid:" << iter->second->meta_position << endl;
                return false;
            }
            break;

        case "nz_col_indices_after_interlance_storage"_hash:
            nz_col_indices_after_interlance_storage[index] = iter->second->get_metadata_arr();
            break;
        case "nz_row_indices_after_interlance_storage"_hash:
            nz_row_indices_after_interlance_storage[index] = iter->second->get_metadata_arr();
            break;
        case "nz_vals_after_interlance_storage"_hash:
            nz_vals_after_interlance_storage[index] = iter->second->get_metadata_arr();
            break;
        default:
            break;
        }
        iter++;
    }

    //遍历所有的sub_matrix
    for (int i = 0; i < num; i++)
    {
        //检查非零元 行和列的index数量是否一致
        if (nz_col_indices[i] != NULL && nz_row_indices[i] != NULL)
        {
            // cout<<"1"<<endl;
            if (nz_col_indices[i]->get_len() != nz_row_indices[i]->get_len())
            {

                cout << "meta_data_set::logical_check(): nz_col_indices length invalid"
                     << " " << i << "/" << (num - 1) << " "
                     << "sub matrix" << endl;
                return false;
            }
        }
        //检查非零元 值和行的index数量是否一致
        if (nz_vals[i] != NULL && nz_row_indices[i] != NULL)
        {
            // cout<<"2"<<endl;
            if (nz_vals[i]->get_len() != nz_row_indices[i]->get_len())
            {
                cout << "meta_data_set::logical_check(): nz_vals length invalid"
                     << " " << i << "/" << (num - 1) << " "
                     << "sub matrix" << endl;
                return false;
            }
        }
        //检查当前矩阵起始和结束行号
        if (begin_row_index[i] != NULL && end_row_index[i] != NULL)
        {
            // cout<<"3"<<endl;
            if (begin_row_index[i]->read_integer_from_arr(0) > end_row_index[i]->read_integer_from_arr(0))
            {
                cout << "meta_data_set::logical_check(): begin_row_index and end_row_index invalid"
                     << " " << i << "/" << (num - 1) << " "
                     << "sub matrix" << endl;
                return false;
            }
        }
        //检查当前矩阵起始和结束列号
        if (begin_col_index[i] != NULL && end_col_index[i] != NULL)
        {
            // cout<<"4"<<endl;
            if (begin_col_index[i]->read_integer_from_arr(0) > end_col_index[i]->read_integer_from_arr(0))
            {
                cout << "meta_data_set::logical_check(): begin_col_index and end_col_index invalid"
                     << " " << i << "/" << (num - 1) << " "
                     << "sub matrix" << endl;
                return false;
            }
        }
        //检查当前矩阵 tblock、warp、thread 级首个非零元的对齐关系
        if (first_nz_indices_tblock[i] != NULL && first_nz_indices_warp[i] != NULL)
        {
            // cout<<"5"<<endl;
            if ((first_nz_indices_tblock[i]->read_integer_from_arr(0) != first_nz_indices_warp[i]->read_integer_from_arr(0)) || (first_nz_indices_tblock[i]->read_integer_from_arr(first_nz_indices_tblock[i]->get_len() - 1) != first_nz_indices_warp[i]->read_integer_from_arr(first_nz_indices_warp[i]->get_len() - 1)))
            {
                cout << "meta_data_set::logical_check(): first_nz_indices invalid in TBLOCK and WARP"
                     << " " << i << "/" << (num - 1) << " "
                     << "sub matrix" << endl;
                return false;
            }
        }

        if (first_nz_indices_warp[i] != NULL && first_nz_indices_thread[i] != NULL)
        {
            // cout<<"6"<<endl;
            if ((first_nz_indices_warp[i]->read_integer_from_arr(0) != first_nz_indices_thread[i]->read_integer_from_arr(0)) || (first_nz_indices_warp[i]->read_integer_from_arr(first_nz_indices_warp[i]->get_len() - 1) != first_nz_indices_thread[i]->read_integer_from_arr(first_nz_indices_thread[i]->get_len() - 1)))
            {
                cout << "meta_data_set::logical_check(): first_nz_indices invalid in THREAD and WARP"
                     << " " << i << "/" << (num - 1) << " "
                     << "sub matrix" << endl;
                return false;
            }
        }
        //检查当前矩阵 tblock、warp 级首个行号的对齐关系
        if (first_row_indices_tblock[i] != NULL && first_row_indices_warp[i] != NULL)
        {
            // cout<<"7"<<endl;
            if ((first_row_indices_tblock[i]->read_integer_from_arr(0) != first_row_indices_warp[i]->read_integer_from_arr(0)) || (first_row_indices_tblock[i]->read_integer_from_arr(first_row_indices_tblock[i]->get_len() - 1) != first_row_indices_warp[i]->read_integer_from_arr(first_row_indices_warp[i]->get_len() - 1)))
            {
                cout << "meta_data_set::logical_check(): first_row_indices invalid in TBLOCK and WARP"
                     << " " << i << "/" << (num - 1) << " "
                     << "sub matrix" << endl;
                return false;
            }
        }
        if (first_row_indices_tblock[i] != NULL && first_nz_indices_tblock[i] != NULL)
        {
            // cout<<"8"<<endl;
            if (first_row_indices_tblock[i]->get_len() != first_nz_indices_tblock[i]->get_len())
            {
                cout << "meta_data_set::logical_check(): first_row_indices or first_nz_indices invalid in TBLOCK"
                     << " " << i << "/" << (num - 1) << " "
                     << "sub matrix" << endl;
                return false;
            }
        }

        if (first_row_indices_tblock_without_ending[i] != NULL && first_nz_indices_tblock[i] != NULL)
        {
            // cout<<"8"<<endl;
            if (first_row_indices_tblock_without_ending[i]->get_len() != first_nz_indices_tblock[i]->get_len() - 1)
            {
                cout << "meta_data_set::logical_check(): first_row_indices_without_ending or first_nz_indices invalid in TBLOCK"
                     << " " << i << "/" << (num - 1) << " "
                     << "sub matrix" << endl;
                return false;
            }
        }
        
        if (first_row_indices_warp[i] != NULL && first_nz_indices_warp[i] != NULL)
        {
            // cout<<"9"<<endl;
            if (first_row_indices_warp[i]->get_len() != first_nz_indices_warp[i]->get_len())
            {
                cout << "meta_data_set::logical_check(): first_row_indices or first_nz_indices invalid in WARP"
                     << " " << i << "/" << (num - 1) << " "
                     << "sub matrix" << endl;
                return false;
            }
        }
        if (first_nz_indices_without_ending[i] != NULL && first_nz_indices_thread[i] != NULL)
        {
            // cout<<"10"<<endl;
            if (first_nz_indices_without_ending[i]->get_len() != first_nz_indices_thread[i]->get_len() - 1)
            {
                cout << "meta_data_set::logical_check(): first_nz_indices_without_ending or first_nz_indices invalid in THREAD"
                     << " " << i << "/" << (num - 1) << " "
                     << "sub matrix" << endl;
                return false;
            }
        }
        if (first_nz_indices_warp[i] != NULL && BMT_size_of_each_blk_warp[i] != NULL)
        {
            // cout<<"11"<<endl;
            if (BMT_size_of_each_blk_warp[i]->get_len() != first_nz_indices_warp[i]->get_len() - 1)
            {
                cout << "meta_data_set::logical_check(): first_nz_indices or BMT_size_of_each_blk invalid in WARP"
                     << " " << i << "/" << (num - 1) << " "
                     << "sub matrix" << endl;
                return false;
            }
        }
        if (first_nz_indices_tblock[i] != NULL && BMT_size_of_each_blk_tblock[i] != NULL)
        {
            // cout<<"12"<<endl;
            if (BMT_size_of_each_blk_tblock[i]->get_len() != first_nz_indices_tblock[i]->get_len() - 1)
            {
                cout << "meta_data_set::logical_check(): first_nz_indices or BMT_size_of_each_blk invalid in TBLOCK"
                     << " " << i << "/" << (num - 1) << " "
                     << "sub matrix" << endl;
                return false;
            }
        }
        if (first_nz_indices_relative_to_BMTB_warp[i] != NULL && first_nz_indices_warp[i] != NULL)
        {
            // cout<<"13"<<endl;
            if (first_nz_indices_relative_to_BMTB_warp[i]->get_len() != first_nz_indices_warp[i]->get_len() - 1)
            {
                cout << "meta_data_set::logical_check(): first_nz_indices or first_nz_indices_relative_to_BMTB invalid in WARP"
                     << " " << i << "/" << (num - 1) << " "
                     << "sub matrix" << endl;
                return false;
            }
        }
        if (first_nz_indices_relative_to_BMTB_thread[i] != NULL && first_nz_indices_thread[i] != NULL)
        {
            // cout<<"13"<<endl;
            if (first_nz_indices_relative_to_BMTB_thread[i]->get_len() != first_nz_indices_thread[i]->get_len() - 1)
            {
                cout << "meta_data_set::logical_check(): first_nz_indices or first_nz_indices_relative_to_BMTB invalid in THREAD"
                     << " " << i << "/" << (num - 1) << " "
                     << "sub matrix" << endl;
                return false;
            }
        }

        if (first_nz_indices_relative_to_BMW[i] != NULL && first_nz_indices_thread[i] != NULL)
        {
            // cout<<"13"<<endl;
            if (first_nz_indices_relative_to_BMW[i]->get_len() != first_nz_indices_thread[i]->get_len() - 1)
            {
                cout << "meta_data_set::logical_check(): first_nz_indices or first_nz_indices_relative_to_BMTB invalid in THREAD"
                     << " " << i << "/" << (num - 1) << " "
                     << "sub matrix" << endl;
                return false;
            }
        }    


        if (first_row_indices_relative_to_BMTB_warp[i] != NULL && first_row_indices_warp[i] != NULL)
        {
            // cout<<"14"<<endl;
            if (first_row_indices_relative_to_BMTB_warp[i]->get_len() != first_row_indices_warp[i]->get_len() - 1)
            {
                cout << "meta_data_set::logical_check(): first_row_indices or first_row_indices_relative_to_BMTB invalid in WARP"
                     << " " << i << "/" << (num - 1) << " "
                     << "sub matrix" << endl;
                return false;
            }
        }

        if (first_row_indices_relative_to_BMTB_warp[i] != NULL && first_row_indices_warp_without_ending[i] != NULL)
        {
            // cout<<"14"<<endl;
            if (first_row_indices_relative_to_BMTB_warp[i]->get_len() != first_row_indices_warp_without_ending[i]->get_len())
            {
                cout << "meta_data_set::logical_check(): first_row_indices or first_row_indices_relative_to_BMTB invalid in WARP"
                     << " " << i << "/" << (num - 1) << " "
                     << "sub matrix" << endl;
                return false;
            }
        }
        if (first_row_indices_relative_to_BMTB_thread[i] != NULL && first_row_indices_thread_without_ending[i] != NULL)
        {
            // cout<<"15"<<endl;
            if (first_row_indices_relative_to_BMTB_thread[i]->get_len() != first_row_indices_thread_without_ending[i]->get_len())
            {
                cout << "meta_data_set::logical_check(): first_row_indices or first_row_indices_relative_to_BMTB invalid in THREAD"
                     << " " << i << "/" << (num - 1) << " "
                     << "sub matrix" << endl;
                return false;
            }
        }

        if (first_row_indices_relative_to_BMTB_thread[i] != NULL && first_row_indices_thread[i] != NULL)
        {
            // cout<<"15"<<endl;
            if (first_row_indices_relative_to_BMTB_thread[i]->get_len() != first_row_indices_thread[i]->get_len() -1)
            {
                cout << "meta_data_set::logical_check(): first_row_indices or first_row_indices_relative_to_BMTB invalid in THREAD"
                     << " " << i << "/" << (num - 1) << " "
                     << "sub matrix" << endl;
                return false;
            }
        }

        if (first_row_indices_relative_to_BMW[i] != NULL && first_row_indices_thread_without_ending[i] != NULL)
        {
            // cout<<"16"<<endl;
            if (first_row_indices_relative_to_BMW[i]->get_len() != first_row_indices_thread_without_ending[i]->get_len())
            {
                cout << "meta_data_set::logical_check(): first_row_indices or first_row_indices_relative_to_BMW invalid in THREAD"
                     << " " << i << "/" << (num - 1) << " "
                     << "sub matrix" << endl;
                return false;
            }
        }

        if (first_row_indices_relative_to_BMW[i] != NULL && first_row_indices_thread[i] != NULL)
        {
            // cout<<"16"<<endl;
            if (first_row_indices_relative_to_BMW[i]->get_len() != first_row_indices_thread[i]->get_len() -1)
            {
                cout << "meta_data_set::logical_check(): first_row_indices or first_row_indices_relative_to_BMW invalid in THREAD"
                     << " " << i << "/" << (num - 1) << " "
                     << "sub matrix" << endl;
                return false;
            }
        }
        if (first_BMT_indices_tblock[i] != NULL && first_nz_indices_tblock[i] != NULL)
        {
            // cout<<"17"<<endl;
            if (first_BMT_indices_tblock[i]->get_len() != first_nz_indices_tblock[i]->get_len())
            {
                cout << "meta_data_set::logical_check(): first_BMT_indices_tblock or first_nz_indices_tblcok invalid"
                     << " " << i << "/" << (num - 1) << " "
                     << "sub matrix" << endl;
                return false;
            }
        }
        if (first_BMT_indices_warp[i] != NULL && first_nz_indices_warp[i] != NULL)
        {
            // cout<<"18"<<endl;
            if (first_BMT_indices_warp[i]->get_len() != first_nz_indices_warp[i]->get_len())
            {
                cout << "meta_data_set::logical_check(): first_BMT_indices_warp or first_nz_indices_warp invalid"
                     << " " << i << "/" << (num - 1) << " "
                     << "sub matrix" << endl;
                return false;
            }
        }
        if (first_row_indices_tblock[i] != NULL && first_row_indices_warp[i] != NULL && first_row_indices_relative_to_BMTB_warp[i] != NULL)
        {
            // cout<<"19"<<endl;
            int k = 0;
            for (int j = 0; j < first_row_indices_relative_to_BMTB_warp[i]->get_len(); j++)
            {
                if (first_row_indices_warp[i]->read_integer_from_arr(j) != first_row_indices_relative_to_BMTB_warp[i]->read_integer_from_arr(j) + first_row_indices_tblock[i]->read_integer_from_arr(k))
                {
                    k += 1;
                }
            }
            if (k+1 != (first_row_indices_tblock[i]->get_len() - 1))
            {
                cout << "meta_data_set::logical_check(): first_row_indices or first_row_indices_relative_to_BMTB invalid in WARP"
                     << " " << i << "/" << (num - 1) << " "
                     << "sub matrix" << endl;
                return false;
            }
        }

        if (first_row_indices_tblock[i] != NULL && first_row_indices_warp_without_ending[i] != NULL && first_row_indices_relative_to_BMTB_warp[i] != NULL)
        {
            // cout<<"19"<<endl;
            int k = 0;
            for (int j = 0; j < first_row_indices_relative_to_BMTB_warp[i]->get_len(); j++)
            {
                if (first_row_indices_warp_without_ending[i]->read_integer_from_arr(j) != first_row_indices_relative_to_BMTB_warp[i]->read_integer_from_arr(j) + first_row_indices_tblock[i]->read_integer_from_arr(k))
                {
                    k += 1;
                }
            }
            if (k+1 != (first_row_indices_tblock[i]->get_len() - 1))
            {
                cout << "meta_data_set::logical_check(): first_row_indices or first_row_indices_relative_to_BMTB invalid in WARP"
                     << " " << i << "/" << (num - 1) << " "
                     << "sub matrix" << endl;
                return false;
            }
        }
        if (first_row_indices_tblock[i] != NULL && first_row_indices_thread_without_ending[i] != NULL && first_row_indices_relative_to_BMTB_thread[i] != NULL)
        {
            // cout<<"20"<<endl;
            int k = 0;
            for (int j = 0; j < first_row_indices_relative_to_BMTB_thread[i]->get_len(); j++)
            {
                if (first_row_indices_thread_without_ending[i]->read_integer_from_arr(j) != first_row_indices_relative_to_BMTB_thread[i]->read_integer_from_arr(j) + first_row_indices_tblock[i]->read_integer_from_arr(k))
                {
                    k += 1;
                }
            }
            if (k+1 != (first_row_indices_tblock[i]->get_len() - 1))
            {
                cout << "meta_data_set::logical_check(): first_row_indices or first_row_indices_relative_to_BMTB invalid in THREAD"
                     << " " << i << "/" << (num - 1) << " "
                     << "sub matrix" << endl;
                return false;
            }
        }

        if (first_row_indices_tblock[i] != NULL && first_row_indices_thread[i] != NULL && first_row_indices_relative_to_BMTB_thread[i] != NULL)
        {
            // cout<<"20"<<endl;
            int k = 0;
            for (int j = 0; j < first_row_indices_relative_to_BMTB_thread[i]->get_len(); j++)
            {
                if (first_row_indices_thread[i]->read_integer_from_arr(j) != first_row_indices_relative_to_BMTB_thread[i]->read_integer_from_arr(j) + first_row_indices_tblock[i]->read_integer_from_arr(k))
                {
                    k += 1;
                }
            }
            if (k+1 != (first_row_indices_tblock[i]->get_len() - 1))
            {
                cout << "meta_data_set::logical_check(): first_row_indices or first_row_indices_relative_to_BMTB invalid in THREAD"
                     << " " << i << "/" << (num - 1) << " "
                     << "sub matrix" << endl;
                return false;
            }
        }

        if (first_row_indices_warp[i] != NULL && first_row_indices_thread_without_ending[i] != NULL && first_row_indices_relative_to_BMW[i] != NULL)
        {
            // cout<<"21"<<endl;
            int k = 0;
            for (int j = 0; j < first_row_indices_relative_to_BMW[i]->get_len(); j++)
            {
                if (first_row_indices_thread_without_ending[i]->read_integer_from_arr(j) != first_row_indices_relative_to_BMW[i]->read_integer_from_arr(j) + first_row_indices_warp[i]->read_integer_from_arr(k))
                {
                    k += 1;
                }
            }
            if (k+1 != (first_row_indices_warp[i]->get_len() - 1))
            {
                cout << "meta_data_set::logical_check(): first_row_indices or first_row_indices_relative_to_BMW invalid in THREAD"
                     << " " << i << "/" << (num - 1) << " "
                     << "sub matrix" << endl;
                return false;
            }
        }

        if (first_row_indices_warp[i] != NULL && first_row_indices_thread[i] != NULL && first_row_indices_relative_to_BMW[i] != NULL)
        {
            // cout<<"21"<<endl;
            int k = 0;
            for (int j = 0; j < first_row_indices_relative_to_BMW[i]->get_len(); j++)
            {
                if (first_row_indices_thread[i]->read_integer_from_arr(j) != first_row_indices_relative_to_BMW[i]->read_integer_from_arr(j) + first_row_indices_warp[i]->read_integer_from_arr(k))
                {
                    k += 1;
                }
            }
            if (k+1 != (first_row_indices_warp[i]->get_len() - 1))
            {
                cout << "meta_data_set::logical_check(): first_row_indices or first_row_indices_relative_to_BMW invalid in THREAD"
                     << " " << i << "/" << (num - 1) << " "
                     << "sub matrix" << endl;
                return false;
            }
        }
        if (first_nz_indices_warp[i] != NULL && first_nz_indices_thread[i] != NULL && first_nz_indices_relative_to_BMW[i] != NULL)
        {
            // cout<<"21"<<endl;
            int k = 0;
            for (int j = 0; j < first_nz_indices_relative_to_BMW[i]->get_len(); j++)
            {
                if (first_nz_indices_thread[i]->read_integer_from_arr(j) != first_nz_indices_relative_to_BMW[i]->read_integer_from_arr(j) + first_nz_indices_warp[i]->read_integer_from_arr(k))
                {
                    k += 1;
                }
            }
            if (k+1 != (first_nz_indices_warp[i]->get_len() - 1))
            {
                cout << "meta_data_set::logical_check(): first_nz_indices or first_nz_indices_relative_to_BMW invalid in THREAD"
                     << " " << i << "/" << (num - 1) << " "
                     << "sub matrix" << endl;
                return false;
            }
        }
        if (first_nz_indices_tblock[i] != NULL && first_nz_indices_warp[i] != NULL && first_nz_indices_relative_to_BMTB_warp[i] != NULL)
        {
            // cout<<"22"<<endl;
            int k = 0;
            for (int j = 0; j < first_row_indices_relative_to_BMTB_warp[i]->get_len(); j++)
            {
                if (first_nz_indices_warp[i]->read_integer_from_arr(j) != first_nz_indices_relative_to_BMTB_warp[i]->read_integer_from_arr(j) + first_nz_indices_tblock[i]->read_integer_from_arr(k))
                {
                    k += 1;
                }
            }
            if (k+1 != (first_nz_indices_tblock[i]->get_len() - 1))
            {
                cout << "meta_data_set::logical_check(): first_nz_indices or first_nz_indices_relative_to_BMTB invalid in WARP"
                     << " " << i << "/" << (num - 1) << " "
                     << "sub matrix" << endl;
                return false;
            }
        }

        if (first_nz_indices_tblock[i] != NULL && first_nz_indices_thread[i] != NULL && first_nz_indices_relative_to_BMTB_thread[i] != NULL)
        {
            // cout<<"22"<<endl;
            int k = 0;
            for (int j = 0; j < first_row_indices_relative_to_BMTB_thread[i]->get_len(); j++)
            {
                if (first_nz_indices_thread[i]->read_integer_from_arr(j) != first_nz_indices_relative_to_BMTB_thread[i]->read_integer_from_arr(j) + first_nz_indices_tblock[i]->read_integer_from_arr(k))
                {
                    k += 1;
                }
            }
            if (k+1 != (first_nz_indices_tblock[i]->get_len() - 1))
            {
                cout << "meta_data_set::logical_check(): first_nz_indices or first_nz_indices_relative_to_BMTB invalid in THREAD"
                     << " " << i << "/" << (num - 1) << " "
                     << "sub matrix" << endl;
                return false;
            }
        }

        if (first_BMT_indices_tblock[i] != NULL && first_row_indices_tblock[i] != NULL && first_row_indices_thread_without_ending[i] != NULL)
        {
            // cout<<"23"<<endl;
            bool no_former_interval_col_direction;
            no_former_interval_col_direction = has_row_direction_blocking_in_specific_level(meta_data_set_ptr, TBLOCK_META, i);
            no_former_interval_col_direction = no_former_interval_col_direction & has_row_direction_blocking_in_specific_level(meta_data_set_ptr, THREAD_META, i);
            if (no_former_interval_col_direction == true)
            {
                for (int j = 0; j < first_row_indices_tblock[i]->get_len() - 1; j++)
                {
                    int k = first_BMT_indices_tblock[i]->read_integer_from_arr(j);
                    if (first_row_indices_tblock[i]->read_integer_from_arr(j) != first_row_indices_thread_without_ending[i]->read_integer_from_arr(k))
                    {
                        cout << "meta_data_set::logical_check(): first_BMT_indices_tblock or first_row_indices_tblock or first_row_indices_without_ending invalid"
                             << " " << i << "/" << (num - 1) << " "
                             << "sub matrix" << endl;
                        return false;
                    }
                }
            }
        }

        if (first_BMT_indices_tblock[i] != NULL && first_row_indices_tblock[i] != NULL && first_row_indices_thread[i] != NULL)
        {
            // cout<<"23"<<endl;
            bool no_former_interval_col_direction;
            no_former_interval_col_direction = has_row_direction_blocking_in_specific_level(meta_data_set_ptr, TBLOCK_META, i);
            no_former_interval_col_direction = no_former_interval_col_direction & has_row_direction_blocking_in_specific_level(meta_data_set_ptr, THREAD_META, i);
            if (no_former_interval_col_direction == true)
            {
                for (int j = 0; j < first_row_indices_tblock[i]->get_len() - 1; j++)
                {
                    int k = first_BMT_indices_tblock[i]->read_integer_from_arr(j);
                    if (first_row_indices_tblock[i]->read_integer_from_arr(j) != first_row_indices_thread[i]->read_integer_from_arr(k))
                    {
                        cout << "meta_data_set::logical_check(): first_BMT_indices_tblock or first_row_indices_tblock or first_row_indices_without_ending invalid"
                             << " " << i << "/" << (num - 1) << " "
                             << "sub matrix" << endl;
                        return false;
                    }
                }
            }
        }
        if (first_BMT_indices_warp[i] != NULL && first_row_indices_warp[i] != NULL && first_row_indices_thread_without_ending[i] != NULL)
        {
            // cout<<"24"<<endl;
            bool no_former_interval_col_direction;
            no_former_interval_col_direction = has_row_direction_blocking_in_specific_level(meta_data_set_ptr, WARP_META, i);
            no_former_interval_col_direction = no_former_interval_col_direction & has_row_direction_blocking_in_specific_level(meta_data_set_ptr, THREAD_META, i);
            if (no_former_interval_col_direction == true)
            {
                for (int j = 0; j < first_row_indices_warp[i]->get_len() - 1; j++)
                {
                    int k = first_BMT_indices_warp[i]->read_integer_from_arr(j);
                    if (first_row_indices_warp[i]->read_integer_from_arr(j) != first_row_indices_thread_without_ending[i]->read_integer_from_arr(k))
                    {
                        cout << "meta_data_set::logical_check(): first_BMT_indices_warp or first_row_indices_warp or first_row_indices_without_ending invalid"
                             << " " << i << "/" << (num - 1) << " "
                             << "sub matrix" << endl;
                        return false;
                    }
                }
            }
        }

        if (first_BMT_indices_warp[i] != NULL && first_row_indices_warp[i] != NULL && first_row_indices_thread[i] != NULL)
        {
            // cout<<"24"<<endl;
            bool no_former_interval_col_direction;
            no_former_interval_col_direction = has_row_direction_blocking_in_specific_level(meta_data_set_ptr, WARP_META, i);
            no_former_interval_col_direction = no_former_interval_col_direction & has_row_direction_blocking_in_specific_level(meta_data_set_ptr, THREAD_META, i);
            if (no_former_interval_col_direction == true)
            {
                for (int j = 0; j < first_row_indices_warp[i]->get_len() - 1; j++)
                {
                    int k = first_BMT_indices_warp[i]->read_integer_from_arr(j);
                    if (first_row_indices_warp[i]->read_integer_from_arr(j) != first_row_indices_thread[i]->read_integer_from_arr(k))
                    {
                        cout << "meta_data_set::logical_check(): first_BMT_indices_warp or first_row_indices_warp or first_row_indices_without_ending invalid"
                             << " " << i << "/" << (num - 1) << " "
                             << "sub matrix" << endl;
                        return false;
                    }
                }
            }
        }

        if (first_BMT_indices_tblock[i] != NULL && first_nz_indices_tblock[i] != NULL && first_nz_indices_thread[i] != NULL)
        {
            // cout<<"25"<<endl;
            for (int j = 0; j < first_nz_indices_tblock[i]->get_len(); j++)
            {
                int k = first_BMT_indices_tblock[i]->read_integer_from_arr(j);
                if (first_nz_indices_tblock[i]->read_integer_from_arr(j) != first_nz_indices_thread[i]->read_integer_from_arr(k))
                {
                    cout << "meta_data_set::logical_check(): first_BMT_indices_tblock or first_nz_indices_tblock or first_nz_indices_thread invalid"
                         << " " << i << "/" << (num - 1) << " "
                         << "sub matrix" << endl;
                    return false;
                }
            }
        }
        if (first_BMT_indices_warp[i] != NULL && first_nz_indices_warp[i] != NULL && first_nz_indices_thread[i] != NULL)
        {
            // cout<<"26"<<endl;
            for (int j = 0; j < first_nz_indices_warp[i]->get_len(); j++)
            {
                int k = first_BMT_indices_warp[i]->read_integer_from_arr(j);
                if (first_nz_indices_warp[i]->read_integer_from_arr(j) != first_nz_indices_thread[i]->read_integer_from_arr(k))
                {
                    cout << j << " " << k << " " << first_nz_indices_warp[i]->read_integer_from_arr(j) << " " << first_nz_indices_thread[i]->read_integer_from_arr(k) << endl;
                    cout << "meta_data_set::logical_check(): first_BMT_indices_warp or first_nz_indices_warp or first_nz_indices_thread invalid"
                         << " " << i << "/" << (num - 1) << " "
                         << "sub matrix" << endl;
                    return false;
                }
            }
        }
        if (first_nz_indices_without_ending[i] != NULL && first_nz_indices_thread[i] != NULL)
        {
            // cout<<"27"<<endl;
            for (int j = 0; j < first_nz_indices_without_ending[i]->get_len(); j++)
            {
                if (first_nz_indices_without_ending[i]->read_integer_from_arr(j) != first_nz_indices_thread[i]->read_integer_from_arr(j))
                {
                    cout << "meta_data_set::logical_check(): first_nz_indices_thread or first_nz_indices_without_ending invalid"
                         << " " << i << "/" << (num - 1) << " "
                         << "sub matrix" << endl;
                    return false;
                }
            }
        }
        if (first_nz_indices_tblock[i] != NULL && BMT_size_of_each_blk_tblock[i] != NULL && first_nz_indices_thread[i] != NULL)
        {
            // cout<<"28"<<endl;
            int nz_id_in_thread = 0;
            int tblock_id = 0;
            int next_tblock_id = 1;
            for (int j = 0; j < first_nz_indices_thread[i]->get_len() - 1; j++)
            {
                if (first_nz_indices_thread[i]->read_integer_from_arr(j) == nz_id_in_thread)
                {
                    nz_id_in_thread += BMT_size_of_each_blk_tblock[i]->read_integer_from_arr(tblock_id);
                    while (nz_id_in_thread == first_nz_indices_tblock[i]->read_integer_from_arr(next_tblock_id))
                    {
                        tblock_id += 1;
                        if (next_tblock_id < first_nz_indices_tblock[i]->get_len() - 1)
                        {
                            next_tblock_id += 1;
                        }
                        if (nz_id_in_thread == first_nz_indices_thread[i]->read_integer_from_arr(first_nz_indices_thread[i]->get_len() - 1))
                        {
                            break;
                        }
                    }
                }
            }

            if (tblock_id != first_nz_indices_tblock[i]->get_len() - 1)
            {
                cout << "meta_data_set::logical_check(): first_nz_indices or BMT_size_of_each_blk_tblock invalid"
                     << " " << i << "/" << (num - 1) << " "
                     << "sub matrix" << endl;
                return false;
            }
        }
        if (first_nz_indices_warp[i] != NULL && BMT_size_of_each_blk_warp[i] != NULL && first_nz_indices_thread[i] != NULL)
        {
            // cout<<"29"<<endl;
            int nz_id_in_thread = 0;
            int warp_id = 0;
            int next_warp_id = 1;
            for (int j = 0; j < first_nz_indices_thread[i]->get_len() - 1; j++)
            {
                if (first_nz_indices_thread[i]->read_integer_from_arr(j) == nz_id_in_thread)
                {
                    nz_id_in_thread += BMT_size_of_each_blk_warp[i]->read_integer_from_arr(warp_id);
                    while (nz_id_in_thread == first_nz_indices_warp[i]->read_integer_from_arr(next_warp_id))
                    {
                        warp_id += 1;
                        if (next_warp_id < first_nz_indices_warp[i]->get_len() - 1)
                        {
                            next_warp_id += 1;
                        }
                        if (nz_id_in_thread == first_nz_indices_thread[i]->read_integer_from_arr(first_nz_indices_thread[i]->get_len() - 1))
                        {
                            break;
                        }
                    }
                }
            }

            if (warp_id != first_nz_indices_warp[i]->get_len() - 1)
            {
                cout << "meta_data_set::logical_check(): first_nz_indices or BMT_size_of_each_blk_warp invalid"
                     << " " << i << "/" << (num - 1) << " "
                     << "sub matrix" << endl;
                return false;
            }
        }

        if (first_BMW_indices_tblock[i] != NULL && first_row_indices_tblock[i] != NULL && first_row_indices_warp[i] != NULL)
        {
            // cout<<"30"<<endl;
            bool no_former_interval_col_direction;
            no_former_interval_col_direction = has_row_direction_blocking_in_specific_level(meta_data_set_ptr, TBLOCK_META, i);
            no_former_interval_col_direction = no_former_interval_col_direction & has_row_direction_blocking_in_specific_level(meta_data_set_ptr, WARP_META, i);
            if (no_former_interval_col_direction == true)
            {
                for (int j = 0; j < first_row_indices_tblock[i]->get_len(); j++)
                {
                    int k = first_BMW_indices_tblock[i]->read_integer_from_arr(j);
                    if (first_row_indices_tblock[i]->read_integer_from_arr(j) != first_row_indices_warp[i]->read_integer_from_arr(k))
                    {
                        cout << "meta_data_set::logical_check(): first_BMW_indices_tblock or first_row_indices_tblock or first_row_indices_warp invalid"
                             << " " << i << "/" << (num - 1) << " "
                             << "sub matrix" << endl;
                        return false;
                    }
                }
            }
        }

        if (first_BMW_indices_tblock[i] != NULL && first_row_indices_tblock[i] != NULL && first_row_indices_warp_without_ending[i] != NULL)
        {
            // cout<<"30"<<endl;
            bool no_former_interval_col_direction;
            no_former_interval_col_direction = has_row_direction_blocking_in_specific_level(meta_data_set_ptr, TBLOCK_META, i);
            no_former_interval_col_direction = no_former_interval_col_direction & has_row_direction_blocking_in_specific_level(meta_data_set_ptr, WARP_META, i);
            if (no_former_interval_col_direction == true)
            {
                for (int j = 0; j < first_row_indices_tblock[i]->get_len(); j++)
                {
                    int k = first_BMW_indices_tblock[i]->read_integer_from_arr(j);
                    if (first_row_indices_tblock[i]->read_integer_from_arr(j) != first_row_indices_warp_without_ending[i]->read_integer_from_arr(k))
                    {
                        cout << "meta_data_set::logical_check(): first_BMW_indices_tblock or first_row_indices_tblock or first_row_indices_warp invalid"
                             << " " << i << "/" << (num - 1) << " "
                             << "sub matrix" << endl;
                        return false;
                    }
                }
            }
        }
        if (first_BMW_indices_tblock[i] != NULL && first_nz_indices_tblock[i] != NULL && first_nz_indices_warp[i] != NULL)
        {
            // cout<<"31"<<endl;
            for (int j = 0; j < first_nz_indices_tblock[i]->get_len(); j++)
            {
                int k = first_BMW_indices_tblock[i]->read_integer_from_arr(j);
                if (first_nz_indices_tblock[i]->read_integer_from_arr(j) != first_nz_indices_warp[i]->read_integer_from_arr(k))
                {
                    cout << "meta_data_set::logical_check(): first_BMW_indices_tblock or first_nz_indices_tblock or first_nz_indices_warp invalid"
                         << " " << i << "/" << (num - 1) << " "
                         << "sub matrix" << endl;
                    return false;
                }
            }
        }
        if (first_BMT_indices_tblock[i] != NULL && first_row_indices_tblock[i] != NULL && first_row_indices_thread_without_ending[i] != NULL)
        {
            // cout<<"32"<<endl;
            for (int j = 0; j < first_row_indices_tblock[i]->get_len() - 1; j++)
            {
                int bmt_id = first_BMT_indices_tblock[i]->read_integer_from_arr(j);
                int next_bmt_id = first_BMT_indices_tblock[i]->read_integer_from_arr(j + 1);
                for (int l = bmt_id; l < next_bmt_id; l++)
                {
                    if (!(first_row_indices_tblock[i]->read_integer_from_arr(j) <= first_row_indices_thread_without_ending[i]->read_integer_from_arr(l) && first_row_indices_thread_without_ending[i]->read_integer_from_arr(l) < first_row_indices_tblock[i]->read_integer_from_arr(j + 1)))
                    {
                        cout << "meta_data_set::logical_check(): first_BMT_indices_tblock or first_row_indices_tblock or first_row_indices_without_ending range invalid"
                             << " " << i << "/" << (num - 1) << " "
                             << "sub matrix" << endl;
                        return false;
                    }
                }
            }
        }

        if (first_BMT_indices_tblock[i] != NULL && first_row_indices_tblock[i] != NULL && first_row_indices_thread[i] != NULL)
        {
            // cout<<"32"<<endl;
            for (int j = 0; j < first_row_indices_tblock[i]->get_len() - 1; j++)
            {
                int bmt_id = first_BMT_indices_tblock[i]->read_integer_from_arr(j);
                int next_bmt_id = first_BMT_indices_tblock[i]->read_integer_from_arr(j + 1);
                for (int l = bmt_id; l < next_bmt_id; l++)
                {
                    if (!(first_row_indices_tblock[i]->read_integer_from_arr(j) <= first_row_indices_thread[i]->read_integer_from_arr(l) && first_row_indices_thread[i]->read_integer_from_arr(l) < first_row_indices_tblock[i]->read_integer_from_arr(j + 1)))
                    {
                        cout << "meta_data_set::logical_check(): first_BMT_indices_tblock or first_row_indices_tblock or first_row_indices_without_ending range invalid"
                             << " " << i << "/" << (num - 1) << " "
                             << "sub matrix" << endl;
                        return false;
                    }
                }
            }
        }

        if (first_BMT_indices_warp[i] != NULL && first_row_indices_warp[i] != NULL && first_row_indices_thread_without_ending[i] != NULL)
        {
            // cout<<"33"<<endl;
            for (int j = 0; j < first_row_indices_warp[i]->get_len() - 1; j++)
            {
                int bmt_id = first_BMT_indices_warp[i]->read_integer_from_arr(j);
                int next_bmt_id = first_BMT_indices_warp[i]->read_integer_from_arr(j + 1);
                for (int l = bmt_id; l < next_bmt_id; l++)
                {
                    if (!(first_row_indices_warp[i]->read_integer_from_arr(j) <= first_row_indices_thread_without_ending[i]->read_integer_from_arr(l) && first_row_indices_thread_without_ending[i]->read_integer_from_arr(l) <= first_row_indices_warp[i]->read_integer_from_arr(j + 1)))
                    {
                        cout << "meta_data_set::logical_check(): first_BMT_indices_warp or first_row_indices_warp or first_row_indices_without_ending range invalid"
                             << " " << i << "/" << (num - 1) << " "
                             << "sub matrix" << endl;
                        return false;
                    }
                }
            }
        }

        if (first_BMT_indices_warp[i] != NULL && first_row_indices_warp[i] != NULL && first_row_indices_thread[i] != NULL)
        {
            // cout<<"33"<<endl;
            for (int j = 0; j < first_row_indices_warp[i]->get_len() - 1; j++)
            {
                int bmt_id = first_BMT_indices_warp[i]->read_integer_from_arr(j);
                int next_bmt_id = first_BMT_indices_warp[i]->read_integer_from_arr(j + 1);
                for (int l = bmt_id; l < next_bmt_id; l++)
                {
                    if (!(first_row_indices_warp[i]->read_integer_from_arr(j) <= first_row_indices_thread[i]->read_integer_from_arr(l) && first_row_indices_thread[i]->read_integer_from_arr(l) <= first_row_indices_warp[i]->read_integer_from_arr(j + 1)))
                    {
                        cout << "meta_data_set::logical_check(): first_BMT_indices_warp or first_row_indices_warp or first_row_indices_without_ending range invalid"
                             << " " << i << "/" << (num - 1) << " "
                             << "sub matrix" << endl;
                        return false;
                    }
                }
            }
        }
        if (first_BMT_indices_tblock[i] != NULL && first_nz_indices_tblock[i] != NULL && first_nz_indices_thread[i] != NULL)
        {
            // cout<<"34"<<endl;
            for (int j = 0; j < first_nz_indices_tblock[i]->get_len() - 1; j++)
            {
                int bmt_id = first_BMT_indices_tblock[i]->read_integer_from_arr(j);
                int next_bmt_id = first_BMT_indices_tblock[i]->read_integer_from_arr(j + 1);
                for (int l = bmt_id; l < next_bmt_id; l++)
                {
                    if (!(first_nz_indices_tblock[i]->read_integer_from_arr(j) <= first_nz_indices_thread[i]->read_integer_from_arr(l) && first_nz_indices_thread[i]->read_integer_from_arr(l) <= first_nz_indices_tblock[i]->read_integer_from_arr(j + 1)))
                    {
                        cout << "meta_data_set::logical_check(): first_BMT_indices_tblock or first_nz_indices_tblock or first_nz_indices_thread range invalid"
                             << " " << i << "/" << (num - 1) << " "
                             << "sub matrix" << endl;
                        return false;
                    }
                }
            }
        }
        if (first_BMT_indices_warp[i] != NULL && first_nz_indices_warp[i] != NULL && first_nz_indices_thread[i] != NULL)
        {
            // cout<<"35"<<endl;
            for (int j = 0; j < first_nz_indices_warp[i]->get_len() - 1; j++)
            {
                int bmt_id = first_BMT_indices_warp[i]->read_integer_from_arr(j);
                int next_bmt_id = first_BMT_indices_warp[i]->read_integer_from_arr(j + 1);
                for (int l = bmt_id; l < next_bmt_id; l++)
                {
                    //行切分时末尾有空行的情况使得nz和下一个父块起始位置相等
                    if (!(first_nz_indices_warp[i]->read_integer_from_arr(j) <= first_nz_indices_thread[i]->read_integer_from_arr(l) && first_nz_indices_thread[i]->read_integer_from_arr(l) <= first_nz_indices_warp[i]->read_integer_from_arr(j + 1)))
                    {
                        cout << j << " "<< l << " " << first_nz_indices_warp[i]->read_integer_from_arr(j) << " " <<first_nz_indices_thread[i]->read_integer_from_arr(l) <<" "<< first_nz_indices_warp[i]->read_integer_from_arr(j + 1) <<endl;
                        cout << "meta_data_set::logical_check(): first_BMT_indices_warp or first_nz_indices_warp or first_nz_indices_thread range invalid"
                             << " " << i << "/" << (num - 1) << " "
                             << "sub matrix" << endl;
                        return false;
                    }
                }
            }
        }
        if (nz_col_indices[i] != NULL && nz_col_indices_after_interlance_storage[i] != NULL)
        {
            if (nz_col_indices[i]->get_len() != nz_col_indices_after_interlance_storage[i]->get_len())
            {
                cout << "meta_data_set::logical_check(): nz_col_indices or nz_col_indices_after_interlance_storage invalid"
                     << " " << i << "/" << (num - 1) << " "
                     << "sub matrix" << endl;
                return false;
            }
        }
        if (nz_row_indices[i] != NULL && nz_row_indices_after_interlance_storage[i] != NULL)
        {
            if (nz_row_indices[i]->get_len() != nz_row_indices_after_interlance_storage[i]->get_len())
            {
                cout << "meta_data_set::logical_check(): nz_row_indices or nz_row_indices_after_interlance_storage invalid"
                     << " " << i << "/" << (num - 1) << " "
                     << "sub matrix" << endl;
                return false;
            }
        }
        if (nz_vals[i] != NULL && nz_vals_after_interlance_storage[i] != NULL)
        {
            if (nz_vals[i]->get_len() != nz_vals_after_interlance_storage[i]->get_len())
            {
                cout << "meta_data_set::logical_check(): nz_vals or nz_vals_after_interlance_storage invalid"
                     << " " << i << "/" << (num - 1) << " "
                     << "sub matrix" << endl;
                return false;
            }
        }
    }
    return true;
}

bool file_is_exist(string file_name)
{
    return (access(file_name.c_str(), F_OK) != -1);
}