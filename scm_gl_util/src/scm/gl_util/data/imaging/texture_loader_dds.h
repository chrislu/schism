
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_UTIL_TEXTURE_LOADER_DDS_H_INCLUDED
#define SCM_GL_UTIL_TEXTURE_LOADER_DDS_H_INCLUDED

#include <scm/core/math.h>
#include <scm/core/numeric_types.h>
#include <scm/core/memory.h>

#include <scm/gl_core/data_formats.h>
#include <scm/gl_core/data_types.h>
#include <scm/gl_core/render_device/render_device_fwd.h>
#include <scm/gl_core/texture_objects/texture_objects_fwd.h>

#include <scm/gl_util/data/imaging/imaging_fwd.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class __scm_export(gl_util) texture_loader_dds
{
public:

    texture_2d_ptr              load_texture_2d(render_device& in_device, const std::string& in_image_path) const;
    texture_3d_ptr              load_texture_3d(render_device& in_device, const std::string& in_image_path) const;

    texture_image_data_ptr      load_image_data(const std::string&  in_image_path) const;

    bool                        save_image_data_dx9(const std::string&           in_image_path,
                                                    const texture_image_data_ptr in_img_data) const;

}; // class texture_loader_dds

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_UTIL_TEXTURE_LOADER_DDS_H_INCLUDED
