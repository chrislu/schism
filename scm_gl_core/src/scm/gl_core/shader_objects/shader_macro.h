
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_SHADER_MACRO_H_INCLUDED
#define SCM_GL_CORE_SHADER_MACRO_H_INCLUDED

#include <string>
#include <vector>

#include <scm/gl_core/shader_objects/shader_objects_fwd.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class __scm_export(gl_core) shader_macro
{
public:
    shader_macro();
    shader_macro(const shader_macro& rhs);
    explicit shader_macro(const std::string& n, const std::string& v);

    shader_macro_array      operator()(const std::string& n, const std::string& v);
    bool                    operator==(const shader_macro& rhs) const;
    bool                    operator!=(const shader_macro& rhs) const;

    std::string             _name;
    std::string             _value;

}; // class shader_macro


class __scm_export(gl_core) shader_macro_array
{
public:
    typedef std::vector<shader_macro> macro_vector;

public:
    shader_macro_array();
    shader_macro_array(const shader_macro_array& rhs);
    shader_macro_array(const shader_macro& in_macro);
    explicit shader_macro_array(const std::string& n, const std::string& v);

    shader_macro_array&     operator()(const shader_macro& in_macro);
    shader_macro_array&     operator()(const std::string& n, const std::string& v);

    size_t                  size() const;
    bool                    empty() const;
    const macro_vector&     macros() const;

    bool                    operator==(const shader_macro_array& rhs) const;
    bool                    operator!=(const shader_macro_array& rhs) const;

private:
    macro_vector            _array;

}; // class shader_macro_array

} // namespace gl
} // namespace scm

#endif // SCM_GL_CORE_SHADER_MACRO_H_INCLUDED
