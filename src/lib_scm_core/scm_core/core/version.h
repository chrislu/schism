
#ifndef VERSION_H_INCLUDED
#define VERSION_H_INCLUDED

#include <string>

#include <scm_core/platform/platform.h>

namespace scm {
namespace core {

static const std::string    SCHISM_NAME         = std::string("schism");

static const unsigned       VERSION_MAJOR       = 0;
static const unsigned       VERSION_MINOR       = 0;
static const unsigned       VERSION_REVISION    = 1;

static const std::string    VERSION_TAG         = std::string("devel");
static const std::string    VERSION_NAME        = std::string("friend of the night");

#if SCM_DEBUG
static const std::string    VERSION_BUILD_TAG   = std::string("debug");
#else
static const std::string    VERSION_BUILD_TAG   = std::string("release");
#endif

// name candidates:
//   - lux aeterna

} // namespace core
} // namespace scm

#endif // VERSION_H_INCLUDED
