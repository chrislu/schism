
#ifndef SCM_GL_CORE_RENDER_DEVICE_H_INCLUDED
#define SCM_GL_CORE_RENDER_DEVICE_H_INCLUDED

#include <iosfwd>
#include <list>
#include <utility>

#include <boost/noncopyable.hpp>
#include <boost/unordered_set.hpp>

#include <scm/core/pointer_types.h>

#include <scm/gl_core/gl_core_fwd.h>
#include <scm/gl_core/buffer.h>
#include <scm/gl_core/shader.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

namespace opengl {
class gl3_core;
} // namespace detail

class __scm_export(gl_core) render_device : boost::noncopyable
{
////// types //////////////////////////////////////////////////////////////////////////////////////
private:
    typedef boost::unordered_set<render_device_resource*>   resource_ptr_set;

    struct shader_macro {
        shader_macro(const std::string& n, const std::string& v) : _name(n), _value(v) {};
        std::string     _name;
        std::string     _value;
    };
    typedef std::list<shader_macro>             shader_macro_list;
    typedef std::list<std::string>              shader_include_list;
    typedef std::list<const shader_ptr>         shader_list;

////// methods ////////////////////////////////////////////////////////////////////////////////////
public:
    render_device();
    virtual ~render_device();

    // device /////////////////////////////////////////////////////////////////////////////////////
    const opengl::gl3_core&         opengl3_api() const;
    render_context_ptr              main_context() const;

    virtual void                    print_device_informations(std::ostream& os) const;

protected:
    void                            register_resource(render_device_resource* res_ptr);
    void                            release_resource(render_device_resource* res_ptr);

    // buffer api /////////////////////////////////////////////////////////////////////////////////
public:
    buffer_ptr                      create_buffer(const buffer::descriptor_type&  buffer_desc,
                                                  const void*                     initial_data);

    // shader api /////////////////////////////////////////////////////////////////////////////////
public:
    void                            add_include_path(const std::string& p);
    void                            add_macro_define(const shader_macro& d);
    shader_ptr                      create_shader(shader::stage_type t, const std::string& s);
    program_ptr                     create_program(const shader_list& in_shaders);
    //shader_ptr                      create_shader(shader::stage_type t, const std::string& s, const shader_macro_list& m, std::ostream& err_os = std::cerr);
    //shader_ptr                      create_shader(shader::stage_type t, const std::string& s, const shader_include_list& i, std::ostream& err_os = std::cerr);
    //shader_ptr                      create_shader(shader::stage_type t, const std::string& s, const shader_macro_list& m, const shader_include_list& i, std::ostream& err_os = std::cerr);
    //shader_ptr                      create_shader_from_file(shader::stage_type t, const std::string& s, std::ostream& err_os = std::cerr);
    //shader_ptr                      create_shader_from_file(shader::stage_type t, const std::string& s, const shader_macro_list& m, std::ostream& err_os = std::cerr);
    //shader_ptr                      create_shader_from_file(shader::stage_type t, const std::string& s, const shader_include_list& i, std::ostream& err_os = std::cerr);
    //shader_ptr                      create_shader_from_file(shader::stage_type t, const std::string& s, const shader_macro_list& m, const shader_include_list& i, std::ostream& err_os = std::cerr);

    //virtual program_ptr                 create_program();
    //shader(shader_type t, const std::string& s, std::ostream& err_os = std::cerr);
    //shader(shader_type t, const std::string& s, const macro_definition_list& m, std::ostream& err_os = std::cerr);
    //shader(shader_type t, const std::string& s, const include_path_list& i, std::ostream& err_os = std::cerr);
    //shader(shader_type t, const std::string& s, const macro_definition_list& m, const include_path_list& i, std::ostream& err_os = std::cerr);

protected:

    // texture api ////////////////////////////////////////////////////////////////////////////////
public:
    //

////// attributes /////////////////////////////////////////////////////////////////////////////////
protected:
    // device /////////////////////////////////////////////////////////////////////////////////////
    shared_ptr<opengl::gl3_core>        _opengl3_api_core;
    render_context_ptr                  _main_context;

    // shader api /////////////////////////////////////////////////////////////////////////////////
    shader_macro_list                   _default_macro_defines;
    shader_include_list                 _default_include_paths;

private:
    resource_ptr_set                    _registered_resources;

}; // class render_device

__scm_export(gl_core) std::ostream& operator<<(std::ostream& os, const render_device& ren_dev);

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_CORE_RENDER_DEVICE_H_INCLUDED
