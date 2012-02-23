
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_GL_CORE_VIEWPORT_H_INCLUDED
#define SCM_GL_CORE_VIEWPORT_H_INCLUDED

#include <vector>

#include <scm/core/math.h>

#include <scm/gl_core/render_device/render_device_fwd.h>
#include <scm/gl_core/render_device/device_resource.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

namespace scm {
namespace gl {

class viewport_array;

class __scm_export(gl_core) viewport
{
public:
    explicit viewport(const math::vec2ui& in_position,
                      const math::vec2ui& in_dimensions,
                      const math::vec2f&  in_depth_range = math::vec2f(0.0f, 1.0f));
    explicit viewport(const math::vec2f& in_position,
                      const math::vec2f& in_dimensions,
                      const math::vec2f& in_depth_range = math::vec2f(0.0f, 1.0f));

    viewport_array      operator()(const math::vec2f& in_position,
                                   const math::vec2f& in_dimensions,
                                   const math::vec2f& in_depth_range = math::vec2f(0.0f, 1.0f));

    bool operator==(const viewport& rhs) const;
    bool operator!=(const viewport& rhs) const;

    math::vec2f     _position;
    math::vec2f     _dimensions;
    math::vec2f     _depth_range;

}; // class viewport

class __scm_export(gl_core) viewport_array
{
public:
    typedef std::vector<viewport> viewport_vector;

public:
    explicit viewport_array(const viewport& in_viewport);
    explicit viewport_array(const math::vec2f& in_position,
                            const math::vec2f& in_dimensions,
                            const math::vec2f& in_depth_range = math::vec2f(0.0f, 1.0f));

    viewport_array&     operator()(const viewport& in_viewport);
    viewport_array&     operator()(const math::vec2f& in_position,
                                   const math::vec2f& in_dimensions,
                                   const math::vec2f& in_depth_range = math::vec2f(0.0f, 1.0f));

    size_t                  size() const;
    const viewport_vector&  viewports() const;

    bool operator==(const viewport_array& rhs) const;
    bool operator!=(const viewport_array& rhs) const;

private:
    viewport_vector     _array;

}; // class viewport_array

} // namespace gl
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_GL_CORE_VIEWPORT_H_INCLUDED
