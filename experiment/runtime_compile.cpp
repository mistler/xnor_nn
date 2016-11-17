#include <fstream>
#include <iostream>
#include <vector>

#include <dlfcn.h>

class Generator {
public:
    Generator() : stream("/tmp/__generated.cpp") {
    }
    void add_func(const std::string &ret_t, const std::string &n,
            const std::string &p, const std::string &b) {
        ret_type.push_back(ret_t);
        name.push_back(n);
        params.push_back(p);
        body.push_back(b);
    }
    int compile() {
        stream << "extern \"C\"{" << std::endl;
        stream << ret_type[0] << " " << name[0] << "(" << params[0] << "){" <<
            body[0] << "}" << std::endl;
        stream << "}" << std::endl;
        system("g++ -g -O0 -c -fpic /tmp/__generated.cpp -o /tmp/__generated.o");
        system("g++ -shared /tmp/__generated.o -o /tmp/__generated.so");
        void *library_handler = dlopen("/tmp/__generated.so", RTLD_LAZY);
        if (!library_handler) throw 1;
        int (*f)(int) = (int (*)(int))dlsym(library_handler, "first");
        return f(13);
    }
    void execute() {}
private:
    std::ofstream stream;
    std::vector<std::string> ret_type, name, params, body;
};

int main() {
    Generator gen;
    gen.add_func("int", "first", "int i", "return i + 5;");
    std::cout << gen.compile() << std::endl;
    return 0;
}
