
#include "shader_object.h"

#include <fstream>
#include <boost/scoped_array.hpp>

#include <ogl/gl.h>

namespace gl
{
    shader_object::shader_object(unsigned int t)
        : _type(t),
          _obj(0)
    {
        _obj = glCreateShader(_type);
    }

    shader_object::~shader_object()
    {
        glDeleteShader(_obj);
    }
    
    void shader_object::add_defines(const std::string& def)
    {
        _def.append("\n");
        _def.append(def);
    }

    void shader_object::add_include_code(const std::string& inc)
    {
        _inc.append("\n");
        _inc.append(inc);
    }

    bool shader_object::add_include_code_from_file(const std::string& filename)
    {
        std::string code;

        if (!get_source_from_file(filename, code)) {
            return (false);
        }

        add_include_code(code);

        return (true);
    }

    void shader_object::set_source_code(const std::string& src)
    {
        _src = src;
    }

    bool shader_object::set_source_code_from_file(const std::string& filename)
    {
        std::string code;

        if (!get_source_from_file(filename, code)) {
            return (false);
        }

        set_source_code(code);

        return (true);
    }

    bool shader_object::compile()
    {
        int compile_state = 0;
        
        if (_obj == 0) {
            _compiler_out = "shader_object::compile():\n - no valid shader object";
            return (false);
        }

        const char* cinc = _inc.c_str();
        const char* cdef = _def.c_str();
        const char* csrc = _src.c_str();

        boost::scoped_array<const char*> csrcptrs(new const char*[3]);
        csrcptrs[0] = cdef;
        csrcptrs[1] = cinc;
        csrcptrs[2] = csrc;

        glShaderSource(_obj, 3, (const GLchar**)(csrcptrs.get()), NULL);

        glCompileShader(_obj);
        glGetShaderiv(_obj, GL_COMPILE_STATUS, &compile_state);

        if (!compile_state) {
            boost::scoped_array<GLchar> compiler_info;
            int                         info_len;

            glGetShaderiv(_obj, GL_INFO_LOG_LENGTH, &info_len);
            compiler_info.reset(new GLchar[info_len]);
            glGetShaderInfoLog(_obj, info_len, NULL, compiler_info.get());

            _compiler_out = std::string(compiler_info.get());

            glDeleteShader(_obj);
            return (false);
        }

        return (true);
    }
    
    bool shader_object::get_source_from_file(const std::string& filename, std::string& out_code)
    {
        std::ifstream               file;
        boost::scoped_array<char>   code;
        std::streamsize             len;

        file.open(filename.c_str(), std::ios::in | std::ios::binary);

        if (!file) {
            return (false);
        }

        file.seekg (0, std::ios::end);
        len = file.tellg();
        file.seekg (0, std::ios::beg);

        code.reset(new char[len+1]);

        file.read(code.get(), len);

        if (!file) {
            // error reading contents of file
            file.close();
            return (false);
        }
        file.close();

        // terminate string to be sure!
        code[len] = '\0';

        out_code = std::string(code.get());

        return (true);
    }

} // namespace gl



