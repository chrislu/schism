
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_UTIL_VOLUME_READER_VGEO_H_INCLUDED
#define SCM_GL_UTIL_VOLUME_READER_VGEO_H_INCLUDED

#include <scm/gl_util/data/volume/volume_reader_blocked.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class __scm_export(gl_util) volume_reader_vgeo : public volume_reader_blocked
{
public:

public:
    volume_reader_vgeo(const std::string& file_path,
                             bool         file_unbuffered = false);
    virtual ~volume_reader_vgeo();

}; // struct volume_reader_vgeo

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // #define SCM_GL_UTIL_VOLUME_READER_VGEO_H_INCLUDED

