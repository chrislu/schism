
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
namespace gl {

class shader_object;

typedef shared_ptr<shader_object>   shader_obj_ptr;

class __scm_export(ogl) shader_compiler : boost::noncopyable
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

    struct shader_define {
        shader_define(const std::string& n, const std::string& v) : _name(n), _value(v) {};
        std::string     _name;
        std::string     _value;
    };

    typedef boost::filesystem::path     include_path;
    typedef std::list<shader_define>    define_list;
    typedef std::list<include_path>     include_path_list;

public:
    shader_compiler();
    virtual ~shader_compiler();

    void                    add_include_path(const std::string& /*p*/);
    void                    add_define(const shader_define& /*d*/);

    shader_obj_ptr          compile(const shader_type           /*t*/,
                                    const std::string&          /*shader_file*/,
                                    const define_list&          /*defines*/,
                                    const include_path_list&    /*includes*/,
                                          std::ostream&         /*out_stream*/);

private:
    include_path_list       _default_include_paths;
    define_list             _default_defines;

    int                     _default_glsl_version;
    shader_profile          _default_glsl_profile;

    int                     _max_include_recursion_depth;

}; // class shader_compiler

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_SHADER_OBJECTS_COMPILER_H_INCLUDED
