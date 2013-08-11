
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "texture_handle.h"

#include <cassert>
#include <exception>
#include <sstream>
#include <stdexcept>

#include <scm/gl_core/log.h>
#include <scm/gl_core/config.h>
#include <scm/gl_core/render_device.h>
#include <scm/gl_core/render_device/opengl/gl_core.h>
#include <scm/gl_core/render_device/opengl/util/assert.h>
#include <scm/gl_core/render_device/opengl/util/binding_guards.h>
#include <scm/gl_core/render_device/opengl/util/constants_helper.h>
#include <scm/gl_core/render_device/opengl/util/data_format_helper.h>
#include <scm/gl_core/render_device/opengl/util/error_helper.h>

#include <scm/gl_core/texture_objects/texture.h>
#include <scm/gl_core/state_objects/sampler_state.h>

namespace scm {
namespace gl {

namespace util {
} // namespace util

texture_handle::texture_handle(
              render_device& in_device,
        const texture&       in_texture,
        const sampler_state& in_sampler)
  : render_device_resource(in_device)
  , _native_handle(0ull)
  , _native_handle_resident(false)
{
    std::stringstream os;
    os << "texture_handle::texture_handle(): ";
    if (!make_resident(in_device, in_texture, in_sampler, os)) {
        glerr() << log::error << os.str();
        state().set(object_state::OS_BAD);
    }
}

texture_handle::~texture_handle()
{
    std::stringstream os;
    os << "texture_handle::~texture_handle(): ";

    if (!make_non_resident(parent_device(), os)) {
        glerr() << log::error << os.str();
    }
}

uint64
texture_handle::native_handle() const
{
    return _native_handle;
}

bool
texture_handle::native_handle_resident() const
{
    return _native_handle_resident;
}

bool
texture_handle::make_resident(
          render_device& in_device,
    const texture&       in_texture,
    const sampler_state& in_sampler,
          std::ostream&  out_stream)
{
    const opengl::gl_core& glapi = in_device.opengl_api();
    util::gl_error         glerror(glapi);

    if (!glapi.extension_ARB_bindless_texture) {
        return false;
    }

    if (   _native_handle
        && _native_handle_resident) {
        if (!make_non_resident(in_device, out_stream)) {
            out_stream << "texture_handle::make_resident() error making current resident handle non-resident (ARB_bindless_texture).";
            return false;
        }
    }

    _native_handle = glapi.glGetTextureSamplerHandleARB(in_texture.object_id(), in_sampler.sampler_id());

    if (glerror || 0ull == _native_handle) {
        out_stream << "texture::make_resident() error getting texture/sampler handle (ARB_bindless_texture): "
                   << glerror.error_string();
        return false;
    }

    glapi.glMakeTextureHandleResidentARB(_native_handle);

    if (glerror) {
        out_stream << "texture::make_resident() error making texture handle resident (ARB_bindless_texture): "
                   << glerror.error_string();
        return false;
    }

    _native_handle_resident = true;

    return true;

}

bool
texture_handle::make_non_resident(
    const render_device& in_device,
          std::ostream&  out_stream)
{
    const opengl::gl_core& glapi = in_device.opengl_api();
    util::gl_error         glerror(glapi);

    if (!glapi.extension_ARB_bindless_texture) {
        return false;
    }

    if (   _native_handle
        && _native_handle_resident) {
        glapi.glMakeTextureHandleNonResidentARB(_native_handle);
        
        if (glerror) {
            out_stream << "texture_handle::make_non_resident() error making texture handle non-resident (ARB_bindless_texture): "
                       << glerror.error_string();
            return false;
        }
        _native_handle_resident = false;
    }

    return true;
}

} // namespace gl
} // namespace scm
