
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_SHADER_OBJECTS_COMPILER_H_INCLUDED
#define SCM_GL_SHADER_OBJECTS_COMPILER_H_INCLUDED

#include <list>
#include <ostream>
#include <string>

#include <boost/noncopyable.hpp>

#include <boost/filesystem/path.hpp>

#include <scm/core/pointer_types.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl_classic {

class shader_object;
class program_object;

typedef shared_ptr<shader_object>   shader_obj_ptr;

class __scm_export(gl_core) shader_compiler : boost::noncopyable
{
public:
    enum shader_type {
        vertex_shader       = 0x01,
        geometry_shader,
        fragment_shader
    };
    enum shader_profile {
        opengl_core,
        opengl_compatibility
    };

    struct macro_definition {
        macro_definition(const std::string& n, const std::string& v) : _name(n), _value(v) {};
        std::string     _name;
        std::string     _value;
    };

    typedef std::list<macro_definition> macro_definition_list;
    typedef std::list<std::string>      include_path_list;

public:
    shader_compiler();
    virtual ~shader_compiler();

    void                    add_include_path(const std::string& /*p*/);
    void                    add_macro_definition(const macro_definition& /*d*/);

    shader_obj_ptr          compile(const shader_type               /*t*/,
                                    const std::string&              /*shader_file*/,
                                    const macro_definition_list&    /*defines*/,
                                    const include_path_list&        /*includes*/,
                                          std::ostream&             /*out_stream*/);

private:
    include_path_list       _default_include_paths;
    macro_definition_list   _default_defines;

    int                     _default_glsl_version;
    shader_profile          _default_glsl_profile;

    int                     _max_include_recursion_depth;

}; // class shader_compiler

} // namespace gl_classic
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_SHADER_OBJECTS_COMPILER_H_INCLUDED
