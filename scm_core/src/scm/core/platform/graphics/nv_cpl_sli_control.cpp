
// Copyright (c) 2012 Christopher Lux <christopherlux@gmail.com>
// Distributed under the Modified BSD License, see license.txt.

#include "nv_cpl_sli_control.h"

#include <cassert>

#include <scm/core/platform/graphics/detail/nv_cpl_data_control.h>

namespace
{
    // from nvPanelAPI.h
    #define NVCPL_API_NUMBER_OF_GPUS                 7    // Graphics card number of GPUs.
    #define NVCPL_API_NUMBER_OF_SLI_GPUS             8    // Graphics card number of SLI GPU clusters available.
    #define NVCPL_API_SLI_MULTI_GPU_RENDERING_MODE   9    // Get/Set SLI multi-GPU redering mode.

    #define NVCPL_API_SLI_ENABLED                    0x10000000  // SLI enabled, when this bit-mask is set.

    static scm::platform::detail::nv_cpl_data_control nv_cpl_data_ctrl;
} // namespace

namespace scm {
namespace platform {

bool nv_cpl_sli_control::open()
{
    return (nv_cpl_data_ctrl.initialize_cpl_control());
}

void nv_cpl_sli_control::close()
{
    nv_cpl_data_ctrl.close_cpl_control();
}

bool nv_cpl_sli_control::is_sli_enabled() const
{
    assert(nv_cpl_data_ctrl._get_data_int != 0);

    long result = 0;

    nv_cpl_data_ctrl._get_data_int(NVCPL_API_SLI_MULTI_GPU_RENDERING_MODE, &result);

    return ((result &  NVCPL_API_SLI_ENABLED) != 0 ? true : false);
}

unsigned nv_cpl_sli_control:: get_number_of_gpus() const
{
    assert(nv_cpl_data_ctrl._get_data_int != 0);

    long result = 0;

    nv_cpl_data_ctrl._get_data_int(NVCPL_API_NUMBER_OF_GPUS, &result);

    return (result);
}

unsigned nv_cpl_sli_control::get_number_of_sli_gpus() const
{
    assert(nv_cpl_data_ctrl._get_data_int != 0);

    long result = 0;

    nv_cpl_data_ctrl._get_data_int(NVCPL_API_NUMBER_OF_SLI_GPUS, &result);

    return (result);
}

nv_cpl_sli_control::sli_mode_t nv_cpl_sli_control::get_active_sli_mode() const
{
    assert(nv_cpl_data_ctrl._get_data_int != 0);

    long result = 0;

    nv_cpl_data_ctrl._get_data_int(NVCPL_API_SLI_MULTI_GPU_RENDERING_MODE, &result);

    if ((result & SLI_AFR) != 0)
        return (SLI_AFR);

    if ((result & SLI_SFR) != 0)
        return (SLI_SFR);

    if ((result & SLI_SINGLE_GPU) != 0)
        return (SLI_SINGLE_GPU);

    return (SLI_UNKNOWN_MODE);
}

bool nv_cpl_sli_control::set_current_sli_mode(sli_mode_t mode)
{
    assert(nv_cpl_data_ctrl._get_data_int != 0);

    return (nv_cpl_data_ctrl._set_data_int(NVCPL_API_SLI_MULTI_GPU_RENDERING_MODE, mode));
}      

} // namespace platform
} // namespace scm
