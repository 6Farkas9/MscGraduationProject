#include "FileReader.h"

std::string FileReader::read_file_content(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file) return "";
    return std::string(
        std::istreambuf_iterator<char>(file),
        std::istreambuf_iterator<char>()
    );
}