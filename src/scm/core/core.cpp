
#include "core.h"

#include <scm/core/root/root_system.h>

#include <cassert>

namespace {

static bool scm_initialized = false;

static scm::scoped_ptr<scm::core::root_system>  scm_root_sys;

} // namespace

namespace scm {

bool initialize()
{
    if (scm_initialized) {

        return (true);
    }

    scm_root_sys.reset(new scm::core::root_system());

    if (!scm_root_sys->initialize()) {
        scm_root_sys.reset();

        return (false);
    }

    scm_initialized = true;
    return (true);
}

bool shutdown()
{
    if (!scm_initialized) {
        return (true);
    }

    assert(scm_root_sys.get() != 0);

    bool ret = scm_root_sys->shutdown();

    scm_root_sys.reset();

    scm_initialized = false;
    return (ret);
}

} // namespace scm
