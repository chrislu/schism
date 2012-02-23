
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef SCM_INPUT_SPACE_NAVIGATOR_DEVICE_H_INCLUDED
#define SCM_INPUT_SPACE_NAVIGATOR_DEVICE_H_INCLUDED

#include <scm/core/numeric_types.h>
#include <scm/core/memory.h>
#include <scm/core/math.h>

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/platform_warning_disable.h>

class space_navigator_impl;

namespace scm {
namespace inp {

class __scm_export(input) space_navigator
{
public:
    space_navigator();
    virtual ~space_navigator();

    void                update();
    void                reset();

    const math::mat4f&  rotation() const;
    const math::mat4f&  translation() const;

protected:

private:
    math::vec3f         _rotation_sensitivity;
    math::vec3f         _translation_sensitivity;

    math::mat4f         _rotation;
    math::mat4f         _translation;

    shared_ptr<space_navigator_impl>   _device;

    friend class space_navigator_impl;
}; // class space_navigator

} // namespace inp
} // namespace scm

#include <scm/core/utilities/platform_warning_enable.h>

#endif // SCM_INPUT_SPACE_NAVIGATOR_DEVICE_H_INCLUDED
