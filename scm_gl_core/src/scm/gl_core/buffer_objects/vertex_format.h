
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_VERTEX_FORMAT_H_INCLUDED
#define SCM_GL_CORE_VERTEX_FORMAT_H_INCLUDED

#include <list>
#include <string>
#include <vector>

#include <scm/core/numeric_types.h>

#include <scm/gl_core/constants.h>
#include <scm/gl_core/data_types.h>
#include <scm/gl_core/data_formats.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class __scm_export(gl_core) vertex_format
{
public:
    struct __scm_export(gl_core) element {
        element(int stream, const std::string& name, data_type type, int stride = 0, interger_handling int_handling = INT_PURE);
        element(int stream, int location,            data_type type, int stride = 0, interger_handling int_handling = INT_PURE);

        bool operator==(const element& e) const;
        bool operator!=(const element& e) const;

        int                 _buffer_stream;
        std::string         _attrib_name;
        int                 _attrib_location;
        data_type           _type;
        int                 _stride;
        interger_handling   _integer_handling;
    }; // struct element
    typedef std::vector<element> element_array;

public:
    vertex_format(const element_array& in_elements);
    vertex_format(const element& in_element);
    vertex_format(int stream, const std::string& name, data_type type, int stride = 0, interger_handling int_handling = INT_PURE);
    vertex_format(int stream, int location,            data_type type, int stride = 0, interger_handling int_handling = INT_PURE);

    /*virtual*/ ~vertex_format();

    vertex_format&  operator()(const element& in_element);
    vertex_format&  operator()(int stream, const std::string& name, data_type type, int stride = 0, interger_handling int_handling = INT_PURE);
    vertex_format&  operator()(int stream, int location,            data_type type, int stride = 0, interger_handling int_handling = INT_PURE);

    const element_array&    elements() const;
    bool                    generic() const;

    bool operator==(const vertex_format& rhs) const;
    bool operator!=(const vertex_format& rhs) const;

protected:
    element_array       _elements;
    bool                _generic;

}; // class vertex_format

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_CORE_VERTEX_FORMAT_H_INCLUDED
