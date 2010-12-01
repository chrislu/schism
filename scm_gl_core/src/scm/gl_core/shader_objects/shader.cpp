
#include "shader.h"

#include <cassert>
#include <string>
#include <sstream>

#include <boost/regex.hpp>
#include <boost/utility.hpp>

#include <scm/core/pointer_types.h>
#include <scm/core/utilities/foreach.h>

#include <scm/gl_core/config.h>
#include <scm/gl_core/render_device/device.h>
#include <scm/gl_core/render_device/opengl/gl3_core.h>
#include <scm/gl_core/render_device/opengl/util/assert.h>
#include <scm/gl_core/render_device/opengl/util/constants_helper.h>
#include <scm/gl_core/render_device/opengl/util/error_helper.h>

namespace  {

class line_string
{
public:
    line_string(scm::size_t n, const std::string& f) : _number(n), _file_name(f) {}
    std::ostream& operator()(std::ostream& os) const {
        if (_file_name == "0") {
            os << "#line " << _number << " " << _file_name;
        }
        else {
            os << "#line " << _number << " \"" << _file_name << "\"";
        }
        return (os);
    }
private:
    scm::size_t         _number;
    const std::string&  _file_name;
};

std::ostream& operator<<(std::ostream& os, const line_string& ln) {
    return (ln(os));
}

} // namespace 

namespace scm {
namespace gl {

shader_stage
shader::type() const
{
    return (_type);
}

const std::string&
shader::info_log() const
{
    return (_info_log);
}

shader::shader(render_device&                  ren_dev,
               shader_stage                    in_type,
               const std::string&              in_src,
               const std::string&              in_src_name,
               const shader_macro_array&       in_macros,
               const shader_include_path_list& in_inc_paths)
  : render_device_child(ren_dev),
    _type(in_type),
    _gl_shader_obj(0)
{
    const opengl::gl3_core& glapi = ren_dev.opengl3_api();
    util::gl_error          glerror(glapi);

    _gl_shader_obj = glapi.glCreateShader(util::gl_shader_types(in_type));
    if (0 == _gl_shader_obj) {
        state().set(object_state::OS_BAD);
    }
    else {
        std::string preprocessed_source;
        if (preprocess_source_string(ren_dev, in_src, in_src_name, in_macros, preprocessed_source)) {
            compile_source_string(ren_dev, preprocessed_source, in_inc_paths);
        }
        else {
            state().set(object_state::OS_ERROR_SHADER_COMPILE);
        }
    }
    
    gl_assert(glapi, leaving shader::shader());
}

shader::~shader()
{
    const opengl::gl3_core& glapi = parent_device().opengl3_api();

    assert(0 != _gl_shader_obj);
    glapi.glDeleteShader(_gl_shader_obj);
    
    gl_assert(glapi, leaving shader::~shader());
}

bool
shader::preprocess_source_string(      render_device&      ren_dev,
                                 const std::string&        in_src,
                                 const std::string&        in_src_name,
                                 const shader_macro_array& in_macros,
                                       std::string&        out_string)
{
    using boost::regex;
    using boost::regex_match;
    using boost::smatch;

    // comment line (//|/*...*/)          1
    regex    comment_line_open_regex("\\s*(//|/\\*.*)?");
    regex    comment_line_close_regex(".*\\*/\\s*");
    // #version xxx xxxxxxx(//|/*...)                1       2    3                4
    regex    version_line_regex("\\s*#\\s*version\\s+(\\d{3})(\\s+([a-zA-Z]*))?\\s*(//|/\\*.*)?");

    std::string         src_name = (    in_src_name.empty()
                                    || !ren_dev.opengl3_api().extension_ARB_shading_language_include
                                    ? "0"
                                    : in_src_name);

    std::stringstream   in_stream(in_src);
    std::stringstream   out_stream;
    std::string         in_line;
    scm::size_t         line_number = 1;

    bool version_line_found   = false;
    bool multi_line_comment   = false;
    bool macro_lines_inserted = false;

    while (std::getline(in_stream, in_line)) {

        if (multi_line_comment) {
            if (regex_match(in_line, comment_line_close_regex)) {
                multi_line_comment = false;
            }
        }

        if (!version_line_found && !multi_line_comment) {
            smatch m;
            if (regex_match(in_line, m, version_line_regex)) {
                version_line_found = true;
                if (m[4] == "/*") {
                    multi_line_comment = true;
                }
            }
            else if (regex_match(in_line, m, comment_line_open_regex)) {
                if (m[1] == "/*") {
                    multi_line_comment = true;
                }
            }
            else {
                _info_log.clear();
                _info_log = src_name + std::string("(0) : error no #version statement found at beginning of source string.");
                return false;
            }
        }

        out_stream << in_line << std::endl;

        if (   !macro_lines_inserted
            &&  version_line_found
            && !multi_line_comment) {
            // write macro definitions
            
            foreach(const shader_macro& m, in_macros.macros()) {
                out_stream << "#define " << m._name << " " << m._value << std::endl;
            }

            out_stream << line_string(line_number + 1, src_name) << std::endl;

            macro_lines_inserted = true;
        }
        
        ++line_number;
    }

    out_string = out_stream.str();

    return true;
}

bool
shader::compile_source_string(      render_device&            ren_dev,
                              const std::string&              in_src,
                              const shader_include_path_list& in_inc_paths)
{
    const opengl::gl3_core& glapi = ren_dev.opengl3_api();
    util::gl_error          glerror(glapi);

    const char* source_string = in_src.c_str();                                                                         gl_assert(glapi, shader::compile_source_string() before glShaderSource);
    glapi.glShaderSource(_gl_shader_obj, 1, reinterpret_cast<const GLchar**>(boost::addressof(source_string)), NULL);   gl_assert(glapi, shader::compile_source_string() before glCompileShader);
    
    if (glapi.extension_ARB_shading_language_include) {
        if (!in_inc_paths.empty()) {
            scoped_array<const char*> paths(new const char*[in_inc_paths.size()]);
            scoped_array<int>         path_lengths(new int[in_inc_paths.size()]);

            size_t path_index = 0;
            foreach(const std::string& s, in_inc_paths) {
                paths[path_index]        = s.c_str();
                path_lengths[path_index] = static_cast<int>(s.length());
                ++path_index;
            }
            glapi.glCompileShaderIncludeARB(_gl_shader_obj,
                                            static_cast<int>(in_inc_paths.size()),
                                            paths.get(),
                                            path_lengths.get());                                                        gl_assert(glapi, shader::compile_source_string() after glCompileShaderIncludeARB);
        }
        else {
            glapi.glCompileShaderIncludeARB(_gl_shader_obj, 0, 0, 0);                                                   gl_assert(glapi, shader::compile_source_string() after glCompileShaderIncludeARB);
        }
    }
    else {
        glapi.glCompileShader(_gl_shader_obj);                                                                          gl_assert(glapi, shader::compile_source_string() after glCompileShader);
    }

    int compile_state = 0;
    glapi.glGetShaderiv(_gl_shader_obj, GL_COMPILE_STATUS, &compile_state);

    if (GL_TRUE != compile_state) {
        state().set(object_state::OS_ERROR_SHADER_COMPILE);
    }

    int info_len = 0;
    glapi.glGetShaderiv(_gl_shader_obj, GL_INFO_LOG_LENGTH, &info_len);
    if (info_len > 10) {
        _info_log.clear();
        _info_log.resize(info_len);

        assert(_info_log.capacity() >= info_len);

        glapi.glGetShaderInfoLog(_gl_shader_obj, info_len, NULL, &_info_log[0]);
    }

    return (GL_TRUE == compile_state);
}

} // namespace gl
} // namespace scm
