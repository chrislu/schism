
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_GL_MODULE_H_INCLUDED
#define SCM_GL_GL_MODULE_H_INCLUDED

#include <set>
#include <string>

#include <boost/unordered_set.hpp>
#include <boost/utility.hpp>

#include <scm/core/utilities/boost_warning_disable.h>
#include <scm/core/utilities/boost_warning_enable.h>

#include <scm/core/utilities/singleton.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {

class core;

namespace gl_classic {

class __scm_export(gl_core) gl_module : boost::noncopyable
{
public:
    struct context_info {
        int             _version_major;
        int             _version_minor;
        int             _version_release;
        std::string     _version_info;

        std::string     _vendor;
        std::string     _renderer;

        std::string     _profile;

        context_info() : _version_major(0), _version_minor(0), _version_release(0) {}
    };

private:
    typedef std::set<std::string>   extension_string_container;
    //typedef boost::unordered_set<std::string>   extension_string_container;

public:
    gl_module();
    virtual ~gl_module();
    
    // core::system interface
    bool                        initialize(core& c);
    bool                        initialize_gl_context();
    bool                        shutdown(core& c);

    const context_info&         context_information() const;

    bool                        is_supported(const std::string& /*ext*/) const;

private:
    bool                        _initialized;
    context_info                _context_info;
    extension_string_container  _supported_extensions;

    friend __scm_export(gl_core) std::ostream& operator<<(std::ostream& out_stream, const gl_module& gl_mod);

}; // class gl_module

typedef singleton<gl_module>    ogl;

} // namespace gl_classic
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_GL_MODULE_H_INCLUDED
