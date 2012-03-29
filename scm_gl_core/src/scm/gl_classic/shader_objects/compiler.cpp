
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "compiler.h"

#include <ostream>
#include <map>
#include <vector>
#include <string>
#include <sstream>

#include <boost/assign/list_of.hpp>
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>

#pragma warning(push)
#pragma warning(disable : 4267) // warning C4267: 'argument' : conversion from 'size_t' to 'unsigned int', possible loss of data
                                // triggered by boost wave...
// configure wave
#ifdef BOOST_WAVE_PREPROCESS_ERROR_MESSAGE_BODY
#undef BOOST_WAVE_PREPROCESS_ERROR_MESSAGE_BODY
#endif // BOOST_WAVE_PREPROCESS_ERROR_MESSAGE_BODY
#define BOOST_WAVE_PREPROCESS_ERROR_MESSAGE_BODY 1

#ifdef BOOST_WAVE_SUPPORT_PRAGMA_MESSAGE
#undef BOOST_WAVE_SUPPORT_PRAGMA_MESSAGE
#endif // BOOST_WAVE_SUPPORT_PRAGMA_MESSAGE
#define BOOST_WAVE_SUPPORT_PRAGMA_MESSAGE 1

#ifdef BOOST_WAVE_EMIT_PRAGMA_DIRECTIVES
#undef BOOST_WAVE_EMIT_PRAGMA_DIRECTIVES
#endif // BOOST_WAVE_EMIT_PRAGMA_DIRECTIVES
#define BOOST_WAVE_EMIT_PRAGMA_DIRECTIVES 1

#ifdef BOOST_WAVE_SUPPORT_PRAGMA_ONCE
#undef BOOST_WAVE_SUPPORT_PRAGMA_ONCE
#endif // BOOST_WAVE_SUPPORT_PRAGMA_ONCE
#define BOOST_WAVE_SUPPORT_PRAGMA_ONCE 1

//#include <boost/wave.hpp>
//#include <boost/wave/preprocessing_hooks.hpp>
//#include <boost/wave/cpplexer/cpp_lex_token.hpp>
//#include <boost/wave/cpplexer/cpp_lex_iterator.hpp>

#pragma warning(pop)

#include <scm/core/numeric_types.h>
#include <scm/core/io/tools.h>
#include <scm/core/utilities/foreach.h>

#include <scm/gl_classic/opengl.h>


namespace {
#if 0
template<typename input_token_type>
class glsl_preprocessing_hook : public boost::wave::context_policies::eat_whitespace<input_token_type>//boost::wave::context_policies::default_preprocessing_hooks
{
public:
    template <typename context_type, typename container_type>
    bool
    found_unknown_directive(context_type& ctx, const container_type& line, container_type& pending) {
        namespace wave = boost::wave;
        typedef typename container_type::const_iterator iterator_type;

        iterator_type   it = line.begin();
        wave::token_id  id = wave::util::impl::skip_whitespace(it, line.end());

        if (id != wave::T_IDENTIFIER) {
            return (false);
        }

        if (it->get_value() == "version") {

            std::string version_macro("__VERSION__=");
            std::string profile_macro;

            // retrieve version
            wave::token_id  version_id = wave::util::impl::skip_whitespace(++it, line.end());
            if (version_id == wave::T_PP_NUMBER) {
                version_macro += it->get_value().c_str();
            }
            else {
                return (false);
            }

            // retrieve profile name
            wave::token_id  profile_id = wave::util::impl::skip_whitespace(++it, line.end());
            if (profile_id == wave::T_IDENTIFIER) {
            }
            else {
                return (false);
            }

            ctx.add_macro_definition(version_macro, true);

            // pass the line on to the glsl compiler
            std::copy(line.begin(), line.end(), std::back_inserter(pending));

            return (true);
        }
        else if (it->get_value() == "extension") {
            std::copy(line.begin(), line.end(), std::back_inserter(pending));
            return (true);
        }

        return (false);
    }
    template <typename context_type, typename token_type, typename container_type>
    bool
    evaluated_conditional_expression(const context_type& ctx, const token_type& directive, const container_type& expression, bool expression_value) {
        namespace wave = boost::wave;
        typedef typename container_type::const_iterator iterator_type;

        for (iterator_type it = expression.begin(); it != expression.end(); ++it) {
            //wave::token_id  id = wave::util::impl::skip_whitespace(it, expression.end());

            if (*it == wave::T_IDENTIFIER) {
                if (it->get_value() == "GL_stuff") {
                    return (false);
                }
            }
        }
        return (false);
    }

}; // class glsl_preprocessing_hook
#endif
} // namespace detail

namespace scm {
namespace gl_classic {

namespace detail {

//static std::map<shader_compiler::shader_type, GLenum> gl_shader_types
//    = boost::assign::map_list_of(shader_compiler::vertex_shader,   GL_VERTEX_SHADER)
//                                (shader_compiler::geometry_shader, GL_GEOMETRY_SHADER)
//                                (shader_compiler::fragment_shader, GL_FRAGMENT_SHADER);
//
//static std::map<shader_compiler::shader_profile, std::string> shader_profile_strings
//    = boost::assign::map_list_of(shader_compiler::opengl_core,          "core")
//                                (shader_compiler::opengl_compatibility, "compatibility");

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

    path    new_path(p);

    if (exists(new_path)) {
        if (!is_directory(new_path)) {
            new_path = new_path.parent_path();
        }
        _default_include_paths.push_back(new_path.string());
    }
}

void
shader_compiler::add_macro_definition(const macro_definition& d)
{
    _default_defines.push_back(d);
}

shader_obj_ptr
shader_compiler::compile(const shader_type           t,
                         const std::string&          shader_file,
                         const macro_definition_list&          defines,
                         const include_path_list&    includes,
                               std::ostream&         log_stream /*= std::cout*/)
{
    using namespace boost::filesystem;

    path            file_path(shader_file);
    std::string     file_name = file_path.filename().string();

    if (!exists(file_path) || is_directory(file_path)) {
        log_stream << "shader_compiler::compile() <error>: "
                   << shader_file << " does not exist or is a directory" << std::endl;
        return (shader_obj_ptr());
    }

    // generate final includes and defines
    include_path_list   combined_includes(includes);        // custom includes take priority, first in list searched first
    combined_includes.insert(combined_includes.end(), _default_include_paths.begin(), _default_include_paths.end());

    macro_definition_list         combined_defines(_default_defines); // custom defines take priority, last in list inserted last into code
    combined_defines.insert(combined_defines.end(), defines.begin(), defines.end());

    // read source file
    std::string source_string;

    if (!scm::io::read_text_file(shader_file, source_string)) {
        log_stream << "shader_compiler::compile() <error>: "
                   << "unable to read file: " << shader_file << std::endl;
        return (shader_obj_ptr());
    }
#if 0
    namespace wave = boost::wave;

    typedef wave::cpplexer::lex_token<> token_type;
    typedef wave::cpplexer::lex_iterator<token_type> lex_iterator_type;

    typedef wave::context<std::string::iterator, lex_iterator_type,
                          wave::iteration_context_policies::load_file_to_string,
                          glsl_preprocessing_hook<lex_iterator_type::token_type> > context_type;

    wave::util::file_position_type current_position;

    context_type ctx(source_string.begin(), source_string.end(), shader_file.c_str());

    ctx.set_language(wave::enable_single_line(ctx.get_language(),             true));
    ctx.set_language(wave::enable_no_character_validation(ctx.get_language(), false));
    ctx.set_language(wave::enable_insert_whitespace(ctx.get_language(),       false));
    ctx.set_language(wave::enable_preserve_comments(ctx.get_language(),       false));
    ctx.set_language(wave::enable_emit_line_directives(ctx.get_language(),    true));
    ctx.set_language(wave::enable_include_guard_detection(ctx.get_language(), true));
    ctx.set_language(wave::enable_emit_pragma_directives(ctx.get_language(),  true));

    context_type::iterator_type first = ctx.begin();
    context_type::iterator_type last = ctx.end();

    while (first != last) {
        try {
            current_position = (*first).get_position();
            std::cout << (*first).get_value() << std::flush;
            ++first;
        }
        catch (wave::cpp_exception const& e) {
        // some preprocessing error
            std::cout 
                << e.file_name() << "(" << e.line_no() << "): "
                << e.description() << std::endl;
            if (!e.is_recoverable())
                return (shader_obj_ptr());
        }
        catch (std::exception const& e) {
        // use last recognized token to retrieve the error position
            std::cout 
                << current_position.get_file() 
                << "(" << current_position.get_line() << "): "
                << "exception caught: " << e.what()
                << std::endl;
            //return (shader_obj_ptr());
        }
        catch (...) {
        // use last recognized token to retrieve the error position
            std::cout 
                << current_position.get_file() 
                << "(" << current_position.get_line() << "): "
                << "unexpected exception caught." << std::endl;
            //return (shader_obj_ptr());
        }
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
#endif
    //out_stream << std::setprecision(6);

#if 0
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
#endif

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

} // namespace gl_classic
} // namespace scm
