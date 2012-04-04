
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_UTIL_TEXTURE_IMAGE_DATA_H_INCLUDED
#define SCM_GL_UTIL_TEXTURE_IMAGE_DATA_H_INCLUDED

#include <vector>

#include <scm/core/math.h>
#include <scm/core/numeric_types.h>
#include <scm/core/memory.h>

#include <scm/gl_core/data_formats.h>
#include <scm/gl_core/data_types.h>

#include <scm/gl_util/data/imaging/imaging_fwd.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class __scm_export(gl_util) texture_image_data
{
public:
    enum data_origin {
        ORIGIN_LOWER_LEFT   = 0x01,
        ORIGIN_UPPER_LEFT
    }; // enum data_origin

    class level {
        math::vec3ui            _size;
        shared_array<uint8>     _data;

    public:
        level(const math::vec3ui& s, const shared_array<uint8>& d) : _data(d), _size(s) {}

        const math::vec3ui&         size() const { return _size; }
        const shared_array<uint8>&  data() const { return _data; }
    }; // struct level

    typedef std::vector<level>  level_vector;

public:
    texture_image_data(const data_origin   img_origin,
                       const data_format   img_format,
                       const level_vector& img_mip_data);
    texture_image_data(const data_origin   img_origin,
                       const data_format   img_format,
                       const int           layers,
                       const level_vector& img_mip_data);
    /*virtual*/ ~texture_image_data();

    const data_origin           origin() const;
    const data_format           format() const;
    const level&                mip_level(const int i) const;
    int                         mip_level_count() const;
    int                         array_layers() const;

    bool                        flip_vertical();

protected:
    data_origin                 _origin;
    data_format                 _format;
    int                         _layers;
    level_vector                _mip_levels;

}; // struct texture_image_data

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_UTIL_TEXTURE_IMAGE_DATA_H_INCLUDED
