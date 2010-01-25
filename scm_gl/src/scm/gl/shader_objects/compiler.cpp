
#include "compiler.h"

#include <ostream>
#include <map>
#include <vector>
#include <string>
#include <sstream>

#include <boost/assign/list_of.hpp>
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>

#include <scm/core/numeric_types.h>
#include <scm/core/io/tools.h>
#include <scm/core/utilities/foreach.h>

#include <scm/gl/opengl.h>

namespace scm {
namespace gl {

namespace detail {

static std::map<shader_compiler::shader_type, GLenum> gl_shader_types
    = boost::assign::map_list_of(shader_compiler::vertex_shader,   GL_VERTEX_SHADER)
                                (shader_compiler::geometry_shader, GL_GEOMETRY_SHADER)
                                (shader_compiler::fragment_shader, GL_FRAGMENT_SHADER);

static std::map<shader_compiler::shader_profile, std::string> shader_profile_strings
    = boost::assign::map_list_of(shader_compiler::opengl_core,          "core")
                                (shader_compiler::opengl_compatibility, "compatibility");

class line_string
{
public:
    line_string(scm::size_t n, const std::string& f) : _number(n), _file_name(f) {}
    std::ostream& operator()(std::ostream& os) const {
        os << "#line " << _number << " \"" << _file_name << "\"" << std::endl;
        return (os);
    }
private:
    scm::size_t         _number;
    const std::string&  _file_name;
};

std::ostream& operator<<(std::ostream& os, const line_string& ln) {
    return (ln(os));
}

} // namespace detail

shader_compiler::shader_compiler()
  : _max_include_recursion_depth(20),
    _default_glsl_version(150),
    _default_glsl_profile(opengl_compatibility)
{
}

shader_compiler::~shader_compiler()
{
}

void
shader_compiler::add_include_path(const std::string& p)
{
    using namespace boost::filesystem;

    path    new_path(p, native);

    if (exists(new_path)) {
        if (!is_directory(new_path)) {
            new_path = new_path.parent_path();
        }
        _default_include_paths.push_back(new_path);
    }
}

void
shader_compiler::add_define(const shader_define& d)
{
    _default_defines.push_back(d);
}

shader_obj_ptr
shader_compiler::compile(const shader_type           t,
                         const std::string&          shader_file,
                         const define_list&          defines,
                         const include_path_list&    includes,
                               std::ostream&         log_stream /*= std::cout*/)
{
    using namespace boost::filesystem;

    path            file_path(shader_file, native);
    std::string     file_name(file_path.filename());

    if (!exists(file_path) || is_directory(file_path)) {
        log_stream << "shader_compiler::compile() <error>: "
                   << shader_file << " does not exist or is a directory" << std::endl;
        return (shader_obj_ptr());
    }

    // generate final includes and defines
    include_path_list   combined_includes(includes);        // custom includes take priority, first in list searched first
    combined_includes.insert(combined_includes.end(), _default_include_paths.begin(), _default_include_paths.end());

    define_list         combined_defines(_default_defines); // custom defines take priority, last in list inserted last into code
    combined_defines.insert(combined_defines.end(), defines.begin(), defines.end());

    // read source file
    std::string source_string;

    if (!scm::io::read_text_file(shader_file, source_string)) {
        log_stream << "shader_compiler::compile() <error>: "
                   << "unable to read file: " << shader_file << std::endl;
        return (shader_obj_ptr());
    }

    // comment line (//|/*...*/)                 1
    boost::regex    comment_line_open_regex("\\s*(//|/\\*.*)?");
    boost::regex    comment_line_close_regex(".*\\*/\\s*");
    // #version xxx xxxxxxx(//|/*...)                       1       2    3                4
    boost::regex    version_line_regex("\\s*#\\s*version\\s+(\\d{3})(\\s+([a-zA-Z]*))?\\s*(//|/\\*.*)?");
    // #include (<|")...("|>)(//|/*...)                     1     2   3          4
    boost::regex    include_line_regex("\\s*#\\s*include\\s+([\"<)(.*)([\">])\\s*(//|/\\*.*)?");

    std::istringstream  in_stream(source_string);
    std::stringstream   out_stream;
    std::string         in_line;
    scm::size_t         line_number = 1;

    bool version_line_found = false;
    bool multi_line_comment = false;
    while (std::getline(in_stream, in_line)) {
        if (multi_line_comment) {
            if (boost::regex_match(in_line, comment_line_close_regex)) {
                out_stream << in_line << std::endl;
                multi_line_comment = false;
            }
        }
        else if (!version_line_found) {
            boost::smatch m;
            if (boost::regex_match(in_line, m, version_line_regex)) {
                out_stream << in_line << std::endl;
                version_line_found = true;
                if (m[4] == "/*") {
                    multi_line_comment = true;
                }
            }
            else if (boost::regex_match(in_line, m, comment_line_open_regex)) {
                out_stream << in_line << std::endl;
                if (m[1] == "/*") {
                    multi_line_comment = true;
                }
            }
            else {
                // this was not a version line and not a comment
                // so we insert the default version string
                out_stream << "#version "
                           << _default_glsl_version << " "
                           << detail::shader_profile_strings[_default_glsl_profile] << std::endl;
                out_stream << detail::line_string(line_number, file_name);
                version_line_found = true;
            }
        }
        ++line_number;
    }


#if 0 // test
    std::vector<std::string> a;

    a.push_back("#version 150");
    a.push_back("#version 150 core");
    a.push_back("#version 150 core//plaaah");
    a.push_back("#version 150 core //plaaah");
    a.push_back("#version 150 core/*plaaah");
    a.push_back("#version 150 core /*plaaah");
    a.push_back(" # version  150   ");
    a.push_back("  #  version    150    core   ");
    a.push_back("   #   version   150     core//  plaaah  ");
    a.push_back("    #    version     150     core     //    plaaah    ");
    a.push_back("    #    version     150     core/*   plaaah   ");
    a.push_back("    #    version     150     core     /*    plaaah    ");

    foreach(const std::string& s, a) {
        boost::smatch m;
        if (boost::regex_search(s, m, version_regex)) {
            out_stream << m[0] << std::endl;
            out_stream << m[1] << " " << m[2] << " " << m[3] << " " << m[4] << " " << m[5] << " " << m[6] << " " << m[7] << std::endl;
        } else {
            out_stream << "error matching: " << s << std::endl;
        }
    }
#endif

    return (shader_obj_ptr());
}

} // namespace gl
} // namespace scm
