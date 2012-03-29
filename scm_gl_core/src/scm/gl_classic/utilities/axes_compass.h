
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef AXES_COMPASS_H_INCLUDED
#define AXES_COMPASS_H_INCLUDED

#include <scm/core/platform/platform.h>
#include <scm/core/utilities/luabind_warning_disable.h>

namespace scm {
namespace gl_classic {

class __scm_export(gl_core) axes_compass
{
public:
    axes_compass();
    virtual ~axes_compass();

    void                render() const;

protected:
private:

}; // class axes_compass

} // namespace gl_classic
} // namespace scm

#include <scm/core/utilities/luabind_warning_enable.h>

#endif // AXES_COMPASS_H_INCLUDED
