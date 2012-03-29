
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#ifndef NV_CPL_SLI_CONTROL_H_INCLUDED
#define NV_CPL_SLI_CONTROL_H_INCLUDED

#include <scm/core/platform/platform.h>

namespace scm {
namespace platform {

class __scm_export(core) nv_cpl_sli_control
{
public:
    typedef enum {
        SLI_AFR               = 0x00000001,
        SLI_SFR               = 0x00000002,
        SLI_SINGLE_GPU        = 0x00000004,

        SLI_UNKNOWN_MODE
    } sli_mode_t;

public:
    bool            open();
    void            close();

    bool            is_sli_enabled() const;
    unsigned        get_number_of_gpus() const;
    unsigned        get_number_of_sli_gpus() const;

    sli_mode_t      get_active_sli_mode() const;
    bool            set_current_sli_mode(sli_mode_t);
                
}; // class nv_cpl_sli_control

} // namespace platform
} // namespace scm

#endif // NV_CPL_SLI_CONTROL_H_INCLUDED
