#ifndef FILE_READER_H
#define FILE_READER_H
#include <string>
#include <fstream>

class FileReader {
public:
    static std::string read_file_content(const std::string& file_path); 
};

#endif