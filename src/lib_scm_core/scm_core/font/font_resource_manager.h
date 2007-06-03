
#ifndef FONT_RESOURCE_MANAGER_H_INCLUDED
#define FONT_RESOURCE_MANAGER_H_INCLUDED

#include <scm_core/font/face.h>
#include <scm_core/resource/resource_manager.h>

#include <scm_core/platform/platform.h>
#include <scm_core/utilities/platform_warning_disable.h>

namespace scm {
namespace font {

class __scm_export font_resource_manager : public res::resource_manager<face>
{
public:
    font_resource_manager();
    virtual ~font_resource_manager();

};

} // namespace font
} // namespace scm

#include <scm_core/utilities/platform_warning_enable.h>

#endif // FONT_RESOURCE_MANAGER_H_INCLUDED
