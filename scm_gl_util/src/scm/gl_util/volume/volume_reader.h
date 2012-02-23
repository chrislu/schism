
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_UTIL_VOLUME_READER_H_INCLUDED
#define SCM_GL_UTIL_VOLUME_READER_H_INCLUDED

#include <string>

#include <scm/core/math.h>
#include <scm/core/numeric_types.h>
#include <scm/core/memory.h>
#include <scm/core/io/io_fwd.h>

#include <scm/gl_core/data_formats.h>
#include <scm/gl_core/data_types.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class __scm_export(gl_util) volume_reader
{
public:

public:
    volume_reader(const std::string& file_path,
                        bool         file_unbuffered = false);
    virtual ~volume_reader();

    const data_format           format() const;
    const math::vec3ui&         dimensions() const;

                                operator bool() const;
    bool                        operator! () const;

    virtual bool                read(const scm::math::vec3ui& o,
                                     const scm::math::vec3ui& s,
                                           void*              d) = 0;

protected:
    math::vec3ui                _dimensions;
    data_format                 _format;

    shared_ptr<io::file>        _file;
    std::string                 _file_path;

}; // struct volume_reader

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // #define #ifndef SCM_GL_UTIL_VOLUME_READER_H_INCLUDED


