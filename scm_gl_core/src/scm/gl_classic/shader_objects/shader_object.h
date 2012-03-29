
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SHADER_OBJECT_H_INCLUDED
#define SHADER_OBJECT_H_INCLUDED

#include <deque>
#include <string>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl_classic {

class program_object;

class __scm_export(gl_core) shader_object
{
public:
    shader_object(unsigned int);
    virtual ~shader_object();

    void                    add_defines(const std::string& /*def*/);
    void                    add_include_code(const std::string& /*inc*/);
    bool                    add_include_code_from_file(const std::string& /*filename*/);
    void                    set_source_code(const std::string& /*src*/);
    bool                    set_source_code_from_file(const std::string& /*filename*/);

    bool                    compile();
    const bool              compiler_output_available() const { return (!_compiler_out.empty()); }
    const std::string&      compiler_output() const { return (_compiler_out); }

protected:

private:
    bool                    get_source_from_file(const std::string& /*filename*/, std::string& /*out_code*/);

    unsigned int            _obj;
    unsigned int            _type;

    // source strings
    std::string             _inc;
    std::string             _def;
    std::string             _src;

    std::string             _compiler_out;

    friend class gl_classic::program_object;

}; // class shader_object

} // namespace gl_classic
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SHADER_OBJECT_H_INCLUDED
