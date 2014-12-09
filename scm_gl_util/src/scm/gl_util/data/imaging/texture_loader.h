
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_UTIL_TEXTURE_LOADER_H_INCLUDED
#define SCM_GL_UTIL_TEXTURE_LOADER_H_INCLUDED

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

class __scm_export(gl_util) texture_loader
{

public:

    texture_2d_ptr              load_texture_2d(render_device&       in_device,
                                                const std::string&   in_image_path,
                                                bool                 in_create_mips,
                                                bool                 in_color_mips  = false,
                                                const data_format    in_force_internal_format = FORMAT_NULL);

    texture_cube_ptr            load_texture_cube(render_device&       in_device,
                                                  const std::string&   in_image_path_px,
                                                  const std::string&   in_image_path_nx,
                                                  const std::string&   in_image_path_py,
                                                  const std::string&   in_image_path_ny,
                                                  const std::string&   in_image_path_pz,
                                                  const std::string&   in_image_path_nz,
                                                  bool                 in_create_mips,
                                                  bool                 in_color_mips  = false,
                                                  const data_format    in_force_internal_format = FORMAT_NULL);

    bool                        load_texture_image(const render_device_ptr& in_device,
                                                   const texture_2d_ptr&    in_texture,
                                                   const std::string&       in_image_path,
                                                   const texture_region&    in_region,
                                                   const unsigned           in_level);

    texture_image_data_ptr      load_image_data(const std::string&  in_image_path);

}; // class texture_loader

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_UTIL_TEXTURE_LOADER_H_INCLUDED
